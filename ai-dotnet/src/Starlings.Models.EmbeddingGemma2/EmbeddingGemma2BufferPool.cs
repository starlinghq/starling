namespace Starlings.Models.EmbeddingGemma2;

/// <summary>
/// The large per-request buffers of one multimodal embedder (pixel values, resized pixels, vision and multimodal graph
/// inputs and outputs, soft tokens), reused across requests and released when the embedder is disposed.
/// </summary>
/// <remarks>
/// <para>
/// Each request rents its buffers for its own exclusive use and returns them when done, so concurrent callers never
/// share a buffer; the pool holds only buffers no request is using. Without it, every image request allocated several
/// arrays above the large object heap threshold and freed them again. The large object heap is not compacted, so its
/// fragmentation and the GC's committed memory moved by 60 to 130 MiB over an hour-long soak
/// (docs/evidence/eg2-image-loaded-soak/). <see cref="System.Buffers.ArrayPool{T}.Shared"/> also drops idle buffers on
/// gen-2 collections, which moved the same memory.
/// </para>
/// <para>
/// Sizes are rounded up to a quarter-octave class (at most 25% above the request), so a buffer serves nearby sizes.
/// A rent takes the smallest free buffer of the element type that fits; when none fits, it allocates a new one and
/// drops the largest free buffer that was too small, so the retained set follows the largest recent requests instead of
/// accumulating. At most <see cref="MaxFreePerType"/> free buffers are kept per element type (the rest are dropped), so
/// what the pool holds is bounded by the buffers that concurrent requests used at the same time. Rented buffers are
/// not cleared: every user writes the part it reads.
/// </para>
/// </remarks>
internal sealed class EmbeddingGemma2BufferPool : IDisposable
{
    private readonly Lock _gate = new();
    private readonly Dictionary<Type, List<Array>> _free = [];
    private readonly bool _enabled;
    private bool _disposed;

    /// <param name="enabled">False allocates every buffer and keeps none (tests compare the two).</param>
    public EmbeddingGemma2BufferPool(bool enabled = true)
    {
        _enabled = enabled;
    }

    /// <summary>Free buffers kept per element type: enough for the buffers of several concurrent requests.</summary>
    internal static int MaxFreePerType { get; } = Math.Max(16, 4 * Environment.ProcessorCount);

    /// <summary>Bytes in the free buffers (tests, diagnostics).</summary>
    internal long FreeBytes
    {
        get
        {
            lock (_gate)
            {
                long bytes = 0;
                foreach (var list in _free.Values)
                {
                    foreach (var a in list)
                    {
                        bytes += Buffer.ByteLength(a);
                    }
                }

                return bytes;
            }
        }
    }

    /// <summary>A buffer of at least <paramref name="minimumLength"/> elements, for the caller's exclusive use until returned.</summary>
    public T[] Rent<T>(int minimumLength)
        where T : unmanaged
    {
        ArgumentOutOfRangeException.ThrowIfNegative(minimumLength);
        if (!_enabled)
        {
            return GC.AllocateUninitializedArray<T>(minimumLength);
        }

        lock (_gate)
        {
            if (!_disposed && _free.TryGetValue(typeof(T), out var list))
            {
                int best = -1;
                int tooSmall = -1;
                for (int i = 0; i < list.Count; i++)
                {
                    int length = list[i].Length;
                    if (length >= minimumLength)
                    {
                        if (best < 0 || length < list[best].Length)
                        {
                            best = i;
                        }
                    }
                    else if (tooSmall < 0 || length > list[tooSmall].Length)
                    {
                        tooSmall = i;
                    }
                }

                if (best >= 0)
                {
                    var found = (T[])list[best];
                    list.RemoveAt(best);
                    return found;
                }

                // The new buffer below replaces the largest one that no longer fits.
                if (tooSmall >= 0)
                {
                    list.RemoveAt(tooSmall);
                }
            }
        }

        return GC.AllocateUninitializedArray<T>(SizeClass(minimumLength));
    }

    /// <summary>Gives a rented buffer back. The caller must not use it afterwards. Null is ignored.</summary>
    public void Return<T>(T[]? array)
        where T : unmanaged
    {
        if (array is null || !_enabled)
        {
            return;
        }

        lock (_gate)
        {
            if (_disposed)
            {
                return;
            }

            if (!_free.TryGetValue(typeof(T), out var list))
            {
                list = [];
                _free[typeof(T)] = list;
            }

            list.Add(array);
            if (list.Count > MaxFreePerType)
            {
                int smallest = 0;
                for (int i = 1; i < list.Count; i++)
                {
                    if (list[i].Length < list[smallest].Length)
                    {
                        smallest = i;
                    }
                }

                list.RemoveAt(smallest);
            }
        }
    }

    /// <summary>Drops every free buffer; buffers returned later are dropped too. Idempotent.</summary>
    public void Dispose()
    {
        lock (_gate)
        {
            _disposed = true;
            _free.Clear();
        }
    }

    /// <summary>
    /// <paramref name="length"/> rounded up to 1.25, 1.5, 1.75 or 2 times a power of two, so at most 25% above it and
    /// never above <see cref="Array.MaxLength"/>. Lengths up to 1,024 are kept as they are.
    /// </summary>
    internal static int SizeClass(int length)
    {
        if (length <= 1024)
        {
            return length;
        }

        // length is in (power, 2 x power]: the classes are 1.25, 1.5, 1.75 and 2 times power.
        int power = (int)(System.Numerics.BitOperations.RoundUpToPowerOf2((uint)length) / 2);
        int step = Math.Max(power / 4, 256);
        long rounded = ((long)length + step - 1) / step * step;
        return (int)Math.Min(rounded, Array.MaxLength);
    }
}
