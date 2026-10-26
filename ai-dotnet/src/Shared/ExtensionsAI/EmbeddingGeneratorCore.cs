// Shared by Starlings.Models.E5.Extensions.AI and Starlings.Models.EmbeddingGemma2.Extensions.AI (compiled into each,
// internal): the parts of their IEmbeddingGenerator implementations that were copies of each other.
using System.Runtime.CompilerServices;
using Microsoft.Extensions.AI;

namespace Starlings.Models.ExtensionsAI;

/// <summary>
/// The request and result handling the text embedding generators share: failures reported through the returned task,
/// the result list with one creation time and the token usage, and the request-option checks that do not depend on the
/// model (model id, raw representation factory, additional property keys).
/// </summary>
internal static class EmbeddingGeneratorCore
{
    /// <summary>
    /// Runs <paramref name="start"/> and returns its task. Nothing is thrown synchronously: a cancellation for
    /// <paramref name="cancellationToken"/> becomes a cancelled task and any other exception a faulted one.
    /// </summary>
    public static Task<GeneratedEmbeddings<Embedding<float>>> Run(Func<Task<GeneratedEmbeddings<Embedding<float>>>> start, CancellationToken cancellationToken)
    {
        try
        {
            return start();
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            return Task.FromCanceled<GeneratedEmbeddings<Embedding<float>>>(cancellationToken);
        }
#pragma warning disable CA1031 // Every failure is reported through the returned task, never thrown synchronously.
        catch (Exception ex)
#pragma warning restore CA1031
        {
            return Task.FromException<GeneratedEmbeddings<Embedding<float>>>(ex);
        }
    }

    /// <summary>
    /// <see cref="Run(Func{Task{GeneratedEmbeddings{Embedding{float}}}}, CancellationToken)"/> on
    /// <paramref name="scheduler"/>: null runs <paramref name="start"/> on the calling thread; otherwise it is queued to
    /// the scheduler (<see cref="TaskCreationOptions.DenyChildAttach"/>) and the returned task completes with it. A token
    /// cancelled before the work starts cancels the task without running it.
    /// </summary>
    public static Task<GeneratedEmbeddings<Embedding<float>>> Run(Func<Task<GeneratedEmbeddings<Embedding<float>>>> start, TaskScheduler? scheduler, CancellationToken cancellationToken) =>
        scheduler is null
            ? Run(start, cancellationToken)
            : Task.Factory.StartNew(() => Run(start, cancellationToken), cancellationToken, TaskCreationOptions.DenyChildAttach, scheduler).Unwrap();

    /// <summary>
    /// Embeds a stream of texts in order: each value is tokenized and validated (<paramref name="tokenize"/>, given the
    /// value and its position) as it arrives; an <see cref="ArgumentException"/> from it other than
    /// <see cref="InputTooLongException"/> is rethrown naming the position, and the tokenized values are grouped exactly as the encoder's batch plan groups an
    /// explicit batch in input order (a group grows while it has fewer than <paramref name="maxRows"/> rows and, with a
    /// budget, its padded size stays within <paramref name="maxCells"/>). Each full group runs as one
    /// <paramref name="embed"/> call (on <paramref name="scheduler"/> when set, else on the enumerating thread) and its
    /// embeddings are yielded in order; the last group runs when the source ends.
    /// </summary>
    public static async IAsyncEnumerable<Embedding<float>> Stream<TInput>(
        IAsyncEnumerable<string> values,
        Func<string, int, TInput> tokenize,
        Func<TInput, int> length,
        int maxRows,
        int? maxCells,
        Func<IReadOnlyList<TInput>, CancellationToken, float[][]> embed,
        Func<float[], float[]> output,
        string modelId,
        TaskScheduler? scheduler,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        var batch = new List<TInput>(maxRows);
        int width = 0;
        int index = 0;
        await foreach (string? value in values.WithCancellation(cancellationToken).ConfigureAwait(false))
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (value is null)
            {
                throw new ArgumentException($"Value {index} is null.", nameof(values));
            }

            TInput input;
            try
            {
                input = tokenize(value, index);
            }
            catch (ArgumentException ex) when (ex is not InputTooLongException)
            {
                // The encoder's own checks number the input within its call (0); report the position in the stream.
                throw new ArgumentException($"Value {index}: {ex.Message}", nameof(values), ex);
            }

            index++;
            int rowLength = length(input);
            int newWidth = Math.Max(width, rowLength);
            if (batch.Count > 0 && (batch.Count >= maxRows || (maxCells is { } budget && (long)(batch.Count + 1) * newWidth > budget)))
            {
                foreach (var embedding in await EmbedGroupAsync(batch, embed, output, modelId, scheduler, cancellationToken).ConfigureAwait(false))
                {
                    yield return embedding;
                }

                batch.Clear();
                newWidth = rowLength;
            }

            batch.Add(input);
            width = newWidth;
        }

        if (batch.Count > 0)
        {
            foreach (var embedding in await EmbedGroupAsync(batch, embed, output, modelId, scheduler, cancellationToken).ConfigureAwait(false))
            {
                yield return embedding;
            }
        }
    }

    private static async Task<Embedding<float>[]> EmbedGroupAsync<TInput>(
        List<TInput> batch, Func<IReadOnlyList<TInput>, CancellationToken, float[][]> embed, Func<float[], float[]> output,
        string modelId, TaskScheduler? scheduler, CancellationToken cancellationToken)
    {
        TInput[] rows = [.. batch];
        float[][] vectors = scheduler is null
            ? embed(rows, cancellationToken)
            : await Task.Factory.StartNew(() => embed(rows, cancellationToken), cancellationToken, TaskCreationOptions.DenyChildAttach, scheduler).ConfigureAwait(false);
        var createdAt = DateTimeOffset.UtcNow;
        var embeddings = new Embedding<float>[vectors.Length];
        for (int i = 0; i < vectors.Length; i++)
        {
            embeddings[i] = new Embedding<float>(output(vectors[i])) { ModelId = modelId, CreatedAt = createdAt };
        }

        return embeddings;
    }

    /// <summary>
    /// The texts of a request in order; a null value fails with <see cref="ArgumentException"/> naming its position.
    /// </summary>
    public static List<string> Texts(IEnumerable<string> values, string prefix = "")
    {
        ArgumentNullException.ThrowIfNull(values);
        var texts = new List<string>(values is ICollection<string> c ? c.Count : 0);
        foreach (string? value in values)
        {
            if (value is null)
            {
                throw new ArgumentException($"Value {texts.Count} is null.", nameof(values));
            }

            texts.Add(prefix.Length == 0 ? value : prefix + value);
        }

        return texts;
    }

    /// <summary>
    /// The result of a batch: one embedding per vector (each passed through <paramref name="output"/>, which may
    /// return the vector itself), all with <paramref name="modelId"/> and one creation time, and the summed token counts
    /// as input and total usage.
    /// </summary>
    public static GeneratedEmbeddings<Embedding<float>> Result(float[][] vectors, ReadOnlySpan<int> tokenCounts, string modelId, Func<float[], float[]> output)
    {
        var createdAt = DateTimeOffset.UtcNow;
        var result = new GeneratedEmbeddings<Embedding<float>>(vectors.Length);
        foreach (float[] vector in vectors)
        {
            result.Add(new Embedding<float>(output(vector)) { ModelId = modelId, CreatedAt = createdAt });
        }

        long tokens = 0;
        foreach (int count in tokenCounts)
        {
            tokens += count;
        }

        result.Usage = new UsageDetails { InputTokenCount = tokens, TotalTokenCount = tokens };
        return result;
    }

    /// <summary>The result of one queued value (micro-batching): its embedding and its token count as usage.</summary>
    public static GeneratedEmbeddings<Embedding<float>> Single(float[] vector, int tokenCount, string modelId)
    {
        var result = new GeneratedEmbeddings<Embedding<float>>(1)
        {
            new Embedding<float>(vector) { ModelId = modelId, CreatedAt = DateTimeOffset.UtcNow },
        };
        result.Usage = new UsageDetails { InputTokenCount = tokenCount, TotalTokenCount = tokenCount };
        return result;
    }

    /// <summary>
    /// Rejects the request options no generator supports: a <see cref="EmbeddingGenerationOptions.ModelId"/> other
    /// than <paramref name="generatorModelId"/>, a <see cref="EmbeddingGenerationOptions.RawRepresentationFactory"/>,
    /// and an <see cref="EmbeddingGenerationOptions.AdditionalProperties"/> key other than <paramref name="propertyName"/>.
    /// Returns the value of <paramref name="propertyName"/> when present.
    /// </summary>
    /// <exception cref="NotSupportedException">An option is not supported.</exception>
    public static bool CheckOptions(EmbeddingGenerationOptions options, string generatorModelId, string propertyName, out object? propertyValue)
    {
        if (options.ModelId is { } modelId && !string.Equals(modelId, generatorModelId, StringComparison.Ordinal))
        {
            throw new NotSupportedException($"EmbeddingGenerationOptions.ModelId = '{modelId}' is not supported: this generator serves only '{generatorModelId}'.");
        }

        if (options.RawRepresentationFactory is not null)
        {
            throw new NotSupportedException("EmbeddingGenerationOptions.RawRepresentationFactory is not supported: this generator has no provider-specific options object.");
        }

        propertyValue = null;
        bool found = false;
        if (options.AdditionalProperties is { Count: > 0 } properties)
        {
            foreach (var (key, value) in properties)
            {
                if (!string.Equals(key, propertyName, StringComparison.Ordinal))
                {
                    throw new NotSupportedException($"EmbeddingGenerationOptions.AdditionalProperties['{key}'] is not supported; the only supported key is '{propertyName}'.");
                }

                propertyValue = value;
                found = true;
            }
        }

        return found;
    }
}
