using System.Text;
using System.Text.Json;

namespace Starlings.Acquisition;

/// <summary>
/// One pinned file of an asset lock: a path inside the snapshot, its size in bytes, its SHA-256 and, optionally, an
/// explicit download address.
/// </summary>
/// <remarks>
/// By default a file is downloaded from the Hub repository and commit of its lock
/// (<c>{Endpoint}/{repository}/resolve/{commit SHA}/{path}</c>). A file with a <see cref="Source"/> is downloaded from
/// that address instead, for assets that are not in the Hub repository, such as an ONNX graph converted and hosted by
/// Starling (a GitHub release asset). Either way the file lands at <see cref="Path"/> in the lock's snapshot
/// directory and must match <see cref="Size"/> and <see cref="Sha256"/>: the digest, not the address, pins the bytes.
/// The Hugging Face access token is never sent to an explicit source.
/// </remarks>
public sealed record ModelAssetFile
{
    /// <summary>Creates a file entry downloaded from the lock's Hub repository and validates it.</summary>
    /// <param name="path">Relative path inside the repository, with forward slashes (for example <c>onnx/model.onnx</c>).</param>
    /// <param name="size">Exact size in bytes.</param>
    /// <param name="sha256">Lowercase hexadecimal SHA-256 of the file (64 characters).</param>
    public ModelAssetFile(string path, long size, string sha256)
        : this(path, size, sha256, null)
    {
    }

    /// <summary>Creates a file entry and validates it.</summary>
    /// <param name="path">Relative path inside the snapshot directory, with forward slashes.</param>
    /// <param name="size">Exact size in bytes.</param>
    /// <param name="sha256">Lowercase hexadecimal SHA-256 of the file (64 characters).</param>
    /// <param name="source">
    /// Explicit download address (absolute <c>https</c> URL, no user information, no fragment), or null to download
    /// from the lock's Hub repository and commit.
    /// </param>
    public ModelAssetFile(string path, long size, string sha256, Uri? source)
    {
        ArgumentNullException.ThrowIfNull(path);
        ArgumentNullException.ThrowIfNull(sha256);
        LockValidation.ValidatePath(path);
        ArgumentOutOfRangeException.ThrowIfNegative(size);
        if (!LockValidation.IsLowerHex(sha256, 64))
        {
            throw new ArgumentException($"sha256 of {path} must be 64 lowercase hexadecimal characters.", nameof(sha256));
        }

        if (source is not null)
        {
            LockValidation.ValidateSource(source, path);
        }

        Path = path;
        Size = size;
        Sha256 = sha256;
        Source = source;
    }

    /// <summary>Relative path inside the snapshot directory, with forward slashes.</summary>
    public string Path { get; }

    /// <summary>Exact size in bytes.</summary>
    public long Size { get; }

    /// <summary>Lowercase hexadecimal SHA-256.</summary>
    public string Sha256 { get; }

    /// <summary>Explicit download address, or null for the lock's Hub repository and commit.</summary>
    public Uri? Source { get; }
}

/// <summary>
/// Where the bytes of a lock's Hub-layout files are hosted when that is not the lock's own repository, for example a
/// repository of converted files: a Hub repository pinned by full commit SHA.
/// </summary>
public sealed record ModelAssetHosting
{
    /// <summary>Creates the hosting entry and validates it.</summary>
    /// <param name="repository">Hub repository id that hosts the files, <c>owner/name</c>.</param>
    /// <param name="revision">Full 40-character commit SHA of that repository.</param>
    public ModelAssetHosting(string repository, string revision)
    {
        ArgumentNullException.ThrowIfNull(repository);
        ArgumentNullException.ThrowIfNull(revision);
        LockValidation.ValidateRepository(repository);
        if (!LockValidation.IsLowerHex(revision, 40))
        {
            throw new ArgumentException(
                $"The hosting revision must be a full 40-character lowercase commit SHA, not a branch or tag name: '{revision}'.", nameof(revision));
        }

        Repository = repository;
        Revision = revision;
    }

    /// <summary>Hub repository id that hosts the files.</summary>
    public string Repository { get; }

    /// <summary>Full commit SHA of <see cref="Repository"/>.</summary>
    public string Revision { get; }
}

/// <summary>
/// An immutable asset lock (plan sections 9 and 10, gate G0): a Hub repository pinned by full commit SHA and the exact
/// files, sizes and digests a model needs. A lock is data, not weights; it is safe to ship in a code package.
/// Files not in the Hub repository (for example a converted ONNX graph hosted elsewhere) carry an explicit
/// <see cref="ModelAssetFile.Source"/>; they still live in, and are verified in, the same snapshot directory.
/// </summary>
public sealed class ModelAssetLock
{
    /// <summary>
    /// Highest version of the JSON lock format this library reads and writes. Version 2 adds the optional per-file
    /// <c>url</c> (<see cref="ModelAssetFile.Source"/>); version 3 adds the optional <c>display_name</c>
    /// (<see cref="DisplayName"/>) and <c>hosting</c> (<see cref="Hosting"/>). <see cref="ToJson"/> writes the lowest
    /// version that holds the lock, so a lock that uses no newer field stays readable by earlier versions of this library.
    /// </summary>
    public const int SchemaVersion = 3;

    /// <summary>The JSON lock format version without explicit sources.</summary>
    public const int HubOnlySchemaVersion = 1;

    /// <summary>The JSON lock format version with per-file explicit sources and no version 3 field.</summary>
    public const int SourcesSchemaVersion = 2;

    /// <summary>Creates a lock and validates it.</summary>
    /// <param name="name">Short human-readable name used in messages (for example <c>multilingual-e5-small</c>).</param>
    /// <param name="repository">Hub repository id, <c>owner/name</c>.</param>
    /// <param name="revision">Full 40-character commit SHA. Branch and tag names are rejected.</param>
    /// <param name="files">Files to acquire; at least one, no duplicate paths.</param>
    public ModelAssetLock(string name, string repository, string revision, IEnumerable<ModelAssetFile> files)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        ArgumentNullException.ThrowIfNull(repository);
        ArgumentNullException.ThrowIfNull(revision);
        ArgumentNullException.ThrowIfNull(files);
        LockValidation.ValidateRepository(repository);
        if (!LockValidation.IsLowerHex(revision, 40))
        {
            throw new ArgumentException(
                $"revision must be a full 40-character lowercase commit SHA, not a branch or tag name: '{revision}'.", nameof(revision));
        }

        var list = files.ToList();
        if (list.Count == 0)
        {
            throw new ArgumentException("A lock needs at least one file.", nameof(files));
        }

        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var file in list)
        {
            ArgumentNullException.ThrowIfNull(file, nameof(files));
            if (!seen.Add(file.Path))
            {
                throw new ArgumentException($"Duplicate file path in lock: {file.Path}.", nameof(files));
            }
        }

        Name = name;
        Repository = repository;
        Revision = revision;
        Files = list.AsReadOnly();
    }

    /// <summary>Short human-readable name used in messages.</summary>
    public string Name { get; }

    /// <summary>Hub repository id, <c>owner/name</c>.</summary>
    public string Repository { get; }

    /// <summary>Full commit SHA.</summary>
    public string Revision { get; }

    /// <summary>Pinned files.</summary>
    public IReadOnlyList<ModelAssetFile> Files { get; }

    /// <summary>
    /// Optional name for people (a product or model name with spaces and capitals); <see cref="Name"/> stays the short
    /// identifier used in messages and tool commands. Not used for any path. JSON format version 3.
    /// </summary>
    public string? DisplayName
    {
        get;
        init
        {
            if (value is not null && (string.IsNullOrWhiteSpace(value) || value.Any(char.IsControl)))
            {
                throw new ArgumentException("DisplayName must be non-blank text without control characters.", nameof(value));
            }

            field = value;
        }
    }

    /// <summary>
    /// Optional hosting repository: when set, files without an explicit <see cref="ModelAssetFile.Source"/> are downloaded
    /// from <c>{Endpoint}/{Hosting.Repository}/resolve/{Hosting.Revision}/{path}</c> instead of the lock's own
    /// <see cref="Repository"/> and <see cref="Revision"/>, which stay the identity of the model: the cache directory,
    /// <see cref="ModelAssetStore.GetSnapshotDirectory"/> and messages use them. JSON format version 3.
    /// </summary>
    public ModelAssetHosting? Hosting { get; init; }

    /// <summary>The repository the Hub-layout files are downloaded from: <see cref="Hosting"/>'s, else <see cref="Repository"/>.</summary>
    public string HostingRepository => Hosting?.Repository ?? Repository;

    /// <summary>The commit the Hub-layout files are downloaded from: <see cref="Hosting"/>'s, else <see cref="Revision"/>.</summary>
    public string HostingRevision => Hosting?.Revision ?? Revision;

    /// <summary>Total size of all files in bytes.</summary>
    public long TotalSize => Files.Sum(f => f.Size);

    /// <summary>Returns the entry for <paramref name="path"/>, or throws when the lock does not contain it.</summary>
    public ModelAssetFile GetFile(string path) =>
        Files.FirstOrDefault(f => string.Equals(f.Path, path, StringComparison.Ordinal))
        ?? throw new KeyNotFoundException($"{Name}: the lock for {Repository}@{Revision} does not contain {path}.");

    /// <summary>Parses a JSON lock (format version <see cref="SchemaVersion"/>).</summary>
    /// <exception cref="FormatException">The document is not a valid lock.</exception>
    public static ModelAssetLock Parse(string json)
    {
        ArgumentNullException.ThrowIfNull(json);
        try
        {
            using var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;
            int version = root.GetProperty("schema_version").GetInt32();
            if (version is < HubOnlySchemaVersion or > SchemaVersion)
            {
                throw new FormatException($"Unsupported lock schema_version {version}; expected {HubOnlySchemaVersion} to {SchemaVersion}.");
            }

            var files = root.GetProperty("files").EnumerateArray().Select(f =>
            {
                Uri? source = null;
                if (f.TryGetProperty("url", out var url))
                {
                    if (version == HubOnlySchemaVersion)
                    {
                        throw new FormatException($"A schema_version {HubOnlySchemaVersion} lock cannot carry file urls; use schema_version {SourcesSchemaVersion} or later.");
                    }

                    source = new Uri(url.GetString()!, UriKind.Absolute);
                }

                return new ModelAssetFile(
                    f.GetProperty("path").GetString()!,
                    f.GetProperty("size").GetInt64(),
                    f.GetProperty("sha256").GetString()!,
                    source);
            }).ToList();
            bool hasDisplayName = root.TryGetProperty("display_name", out var displayElement);
            bool hasHosting = root.TryGetProperty("hosting", out var hostingElement);
            if ((hasDisplayName || hasHosting) && version < SchemaVersion)
            {
                throw new FormatException($"Only a schema_version {SchemaVersion} lock can carry display_name or hosting.");
            }

            string? displayName = hasDisplayName ? displayElement.GetString() ?? throw new FormatException("display_name must be a string.") : null;
            ModelAssetHosting? hosting = hasHosting
                ? new ModelAssetHosting(hostingElement.GetProperty("repository").GetString()!, hostingElement.GetProperty("revision").GetString()!)
                : null;
            return new ModelAssetLock(
                root.GetProperty("name").GetString()!,
                root.GetProperty("repository").GetString()!,
                root.GetProperty("revision").GetString()!,
                files)
            {
                DisplayName = displayName,
                Hosting = hosting,
            };
        }
        catch (Exception ex) when (ex is JsonException or KeyNotFoundException or InvalidOperationException or ArgumentException or UriFormatException)
        {
            throw new FormatException("Invalid model asset lock: " + ex.Message, ex);
        }
    }

    /// <summary>Reads and parses a JSON lock file.</summary>
    public static ModelAssetLock Load(string path) => Parse(File.ReadAllText(path));

    /// <summary>
    /// Serializes the lock as indented JSON in the lowest format version that holds it: <see cref="SchemaVersion"/> (3)
    /// when <see cref="DisplayName"/> or <see cref="Hosting"/> is set, else <see cref="SourcesSchemaVersion"/> (2) when a
    /// file has an explicit source (a <c>url</c> on those files), else <see cref="HubOnlySchemaVersion"/> (1).
    /// </summary>
    public string ToJson()
    {
        using var buffer = new MemoryStream();
        using (var writer = new Utf8JsonWriter(buffer, new JsonWriterOptions { Indented = true }))
        {
            writer.WriteStartObject();
            int version = DisplayName is not null || Hosting is not null ? SchemaVersion
                : Files.Any(f => f.Source is not null) ? SourcesSchemaVersion
                : HubOnlySchemaVersion;
            writer.WriteNumber("schema_version", version);
            writer.WriteString("name", Name);
            if (DisplayName is not null)
            {
                writer.WriteString("display_name", DisplayName);
            }

            writer.WriteString("repository", Repository);
            writer.WriteString("revision", Revision);
            if (Hosting is not null)
            {
                writer.WriteStartObject("hosting");
                writer.WriteString("repository", Hosting.Repository);
                writer.WriteString("revision", Hosting.Revision);
                writer.WriteEndObject();
            }

            writer.WriteStartArray("files");
            foreach (var file in Files)
            {
                writer.WriteStartObject();
                writer.WriteString("path", file.Path);
                writer.WriteNumber("size", file.Size);
                writer.WriteString("sha256", file.Sha256);
                if (file.Source is not null)
                {
                    writer.WriteString("url", file.Source.AbsoluteUri);
                }

                writer.WriteEndObject();
            }

            writer.WriteEndArray();
            writer.WriteEndObject();
        }

        return Encoding.UTF8.GetString(buffer.ToArray()) + "\n";
    }

    /// <inheritdoc />
    public override string ToString() => $"{Name} ({Repository}@{Revision}, {Files.Count} files)";
}

internal static class LockValidation
{
    public static bool IsLowerHex(string value, int length)
    {
        if (value.Length != length)
        {
            return false;
        }

        foreach (char c in value)
        {
            if (!char.IsAsciiDigit(c) && c is not (>= 'a' and <= 'f'))
            {
                return false;
            }
        }

        return true;
    }

    public static void ValidateRepository(string repository)
    {
        var parts = repository.Split('/');
        if (parts.Length != 2 || parts.Any(p => p.Length == 0 || p is "." or ".." || !p.All(IsNameChar)))
        {
            throw new ArgumentException($"repository must be 'owner/name' using letters, digits, '-', '_' and '.': '{repository}'.", nameof(repository));
        }
    }

    /// <summary>Rejects absolute paths, backslashes, empty or dot segments, so a lock can never write outside its snapshot directory.</summary>
    public static void ValidatePath(string path)
    {
        if (path.Length == 0 || path.Contains('\\', StringComparison.Ordinal) || path.Contains(':', StringComparison.Ordinal) || path.StartsWith('/'))
        {
            throw new ArgumentException($"File path must be relative with forward slashes: '{path}'.", nameof(path));
        }

        foreach (var segment in path.Split('/'))
        {
            if (segment.Length == 0 || segment is "." or ".." || segment.Any(c => char.IsControl(c)))
            {
                throw new ArgumentException($"File path has an empty, '.', '..' or control-character segment: '{path}'.", nameof(path));
            }
        }
    }

    /// <summary>
    /// An explicit source must be an absolute https URL without user information or fragment, so a lock never carries
    /// credentials and never downloads in clear text.
    /// </summary>
    public static void ValidateSource(Uri source, string path)
    {
        if (!source.IsAbsoluteUri || source.Scheme != Uri.UriSchemeHttps || source.UserInfo.Length > 0 || source.Fragment.Length > 0 || source.Host.Length == 0)
        {
            throw new ArgumentException($"The source of {path} must be an absolute https URL without user information or fragment: '{source}'.", nameof(source));
        }
    }

    private static bool IsNameChar(char c) => char.IsAsciiLetterOrDigit(c) || c is '-' or '_' or '.';
}
