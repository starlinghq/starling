using System.Buffers.Binary;
using System.Runtime.InteropServices;

namespace Starlings.Interventions.Internal;

/// <summary>
/// Reads a file's device, inode and modification time, as Python's <c>os.stat</c> reports them. .NET has no public
/// API for the inode, so on Linux this calls <c>statx(2)</c>, whose <c>struct statx</c> layout is the same on every
/// architecture (unlike <c>struct stat</c>). Elsewhere only the modification time is available.
/// </summary>
internal static partial class FileIdentityProbe
{
    private const int AtFdCwd = -100;
    private const uint StatxIno = 0x100;
    private const uint StatxMtime = 0x40;
    private const int StatxSize = 256;

    // struct statx offsets (include/uapi/linux/stat.h).
    private const int MaskOffset = 0;
    private const int InodeOffset = 32;
    private const int MtimeSecondsOffset = 112;
    private const int MtimeNanosecondsOffset = 120;
    private const int DevMajorOffset = 136;
    private const int DevMinorOffset = 140;

    private static int s_statxUnavailable;

    /// <summary>
    /// The identity of <paramref name="path"/> (symlinks followed), or null when this platform cannot report device
    /// and inode (not Linux, or no <c>statx</c> in the C library); then use <see cref="ModifiedNanoseconds"/> alone.
    /// </summary>
    /// <exception cref="FileNotFoundException">The file does not exist.</exception>
    /// <exception cref="IOException">statx failed for another reason.</exception>
    public static InterventionFileIdentity? TryGet(string path)
    {
        if (!OperatingSystem.IsLinux() || Volatile.Read(ref s_statxUnavailable) != 0)
        {
            return null;
        }

        byte[] buffer = new byte[StatxSize];
        int result;
        try
        {
            result = Statx(AtFdCwd, path, 0, StatxIno | StatxMtime, ref buffer[0]);
        }
        catch (Exception e) when (e is DllNotFoundException or EntryPointNotFoundException)
        {
            Volatile.Write(ref s_statxUnavailable, 1);
            return null;
        }

        if (result != 0)
        {
            int errno = Marshal.GetLastPInvokeError();
            if (errno == 38)
            {
                // ENOSYS: a kernel (or seccomp filter) without statx.
                Volatile.Write(ref s_statxUnavailable, 1);
                return null;
            }

            if (errno == 2)
            {
                throw new FileNotFoundException($"Intervention artifact not found: {path}", path);
            }

            throw new IOException($"statx({path}) failed with errno {errno}.");
        }

        ReadOnlySpan<byte> span = buffer;
        uint mask = BinaryPrimitives.ReadUInt32LittleEndian(span[MaskOffset..]);
        if ((mask & (StatxIno | StatxMtime)) != (StatxIno | StatxMtime))
        {
            return null;
        }

        ulong inode = MemoryMarshal.Read<ulong>(span[InodeOffset..]);
        long seconds = MemoryMarshal.Read<long>(span[MtimeSecondsOffset..]);
        uint nanoseconds = MemoryMarshal.Read<uint>(span[MtimeNanosecondsOffset..]);
        uint major = MemoryMarshal.Read<uint>(span[DevMajorOffset..]);
        uint minor = MemoryMarshal.Read<uint>(span[DevMinorOffset..]);
        return new InterventionFileIdentity(
            unchecked((long)MakeDevice(major, minor)),
            unchecked((long)inode),
            checked((seconds * 1_000_000_000L) + nanoseconds));
    }

    /// <summary>Modification time in nanoseconds since the Unix epoch, at the precision .NET reports (100 ns).</summary>
    public static long ModifiedNanoseconds(string path)
    {
        FileInfo info = new(path);
        if (!info.Exists)
        {
            throw new FileNotFoundException($"Intervention artifact not found: {path}", path);
        }

        return (info.LastWriteTimeUtc - DateTime.UnixEpoch).Ticks * 100;
    }

    /// <summary>glibc's <c>gnu_dev_makedev</c>: the <c>st_dev</c> encoding Python's <c>os.stat</c> reports.</summary>
    internal static ulong MakeDevice(uint major, uint minor) =>
        ((ulong)(major & 0xfffff000u) << 32) | ((ulong)(major & 0xfffu) << 8) |
        ((ulong)(minor & 0xffffff00u) << 12) | (minor & 0xffu);

    [LibraryImport("libc", EntryPoint = "statx", SetLastError = true, StringMarshalling = StringMarshalling.Utf8)]
    private static partial int Statx(int dirfd, string path, int flags, uint mask, ref byte buffer);
}
