using System.Runtime.CompilerServices;
using System.Threading.Channels;
using Starlings.Generation.OnnxGenAI;
using Microsoft.Extensions.AI;

namespace Starlings.Models.Qwen35.Extensions.AI;

/// <summary>
/// <c>Qwen/Qwen3.5-0.8B</c> as a Microsoft.Extensions.AI <see cref="IChatClient"/>, over <see cref="Qwen35ChatModel"/>:
/// the same template, tokenizer, decode loop and tokens as the typed API.
/// </summary>
/// <remarks>
/// <para>
/// <b>Messages.</b> <see cref="ChatRole.System"/> (first message only, as the template requires), <see cref="ChatRole.User"/>
/// and <see cref="ChatRole.Assistant"/> messages of <see cref="TextContent"/> (several are concatenated); an assistant
/// message may also carry <see cref="TextReasoningContent"/>, rendered as that turn's reasoning. Tool messages, function
/// calls and results, images and other content, and author names fail with <see cref="NotSupportedException"/>.
/// <see cref="ChatOptions.Instructions"/> becomes the system message when the conversation has none.
/// </para>
/// <para>
/// <b>Options.</b> <see cref="ChatOptions.MaxOutputTokens"/> maps to <see cref="Qwen35ChatOptions.MaxNewTokens"/> and
/// <see cref="ChatOptions.StopSequences"/> to <see cref="Qwen35ChatOptions.StopSequences"/>, each replacing
/// <see cref="Qwen35ChatClientOptions.Defaults"/>. Decoding is greedy unless <see cref="ChatOptions.Temperature"/> (above 0),
/// <see cref="ChatOptions.TopP"/> or <see cref="ChatOptions.TopK"/> is set; sampling then requires
/// <see cref="ChatOptions.Seed"/> (or a seeded default) and fails with <see cref="NotSupportedException"/> without one.
/// <c>Temperature = 0</c> selects greedy decoding. <see cref="ChatOptions.AdditionalProperties"/> may contain only
/// <see cref="EnableThinkingPropertyName"/>. Tools, a non-text <see cref="ChatOptions.ResponseFormat"/>,
/// <see cref="ChatOptions.ConversationId"/>, <see cref="ChatOptions.Reasoning"/>, frequency and presence penalties, background
/// responses, another <see cref="ChatOptions.ModelId"/> and a <see cref="ChatOptions.RawRepresentationFactory"/> fail with
/// <see cref="NotSupportedException"/> rather than being ignored.
/// </para>
/// <para>
/// <b>Responses.</b> One assistant message with the completion text (when thinking is enabled, the reasoning,
/// <c>&lt;/think&gt;</c> and the answer, unsplit). <see cref="ChatResponse.FinishReason"/>: <see cref="ChatFinishReason.Stop"/>
/// for a stop token or a stop sequence, <see cref="ChatFinishReason.Length"/> at the token limit.
/// <see cref="ChatResponse.Usage"/>: input tokens are the prompt ids fed to the model, output tokens the completion ids
/// returned (the stop token excluded; tokens of a matched stop sequence included). <see cref="ChatResponse.RawRepresentation"/>
/// is the <see cref="GenerationResult"/>.
/// </para>
/// <para>
/// <b>Streaming.</b> One <see cref="ChatResponseUpdate"/> per decoded text delta (never half a character: the owned
/// incremental detokenizer holds back incomplete UTF-8 sequences and possible stop-sequence starts). The last update
/// carries any remaining text, the finish reason and a <see cref="UsageContent"/> equal to the non-streamed usage. All
/// updates of a response share its <see cref="ChatResponseUpdate.ResponseId"/>, <see cref="ChatResponseUpdate.MessageId"/>
/// and <see cref="ChatResponseUpdate.CreatedAt"/>; their text concatenates to the non-streamed text for the same request.
/// </para>
/// <para>
/// <b>Execution and threads.</b> Each request runs on its own dedicated background thread (not a thread-pool thread):
/// every prefill chunk and decode step is a blocking native ONNX Runtime GenAI call, so nothing here is asynchronous I/O;
/// the methods are asynchronous only in that the caller's thread and synchronization context are never blocked. One
/// thread per request keeps the native generator on the thread that created it for its whole life, as in the
/// synchronous API that P-016 tested, and avoids blocking thread-pool threads for seconds. The streaming worker runs
/// ahead of the consumer by at most <see cref="Qwen35ChatClientOptions.StreamingBufferCapacity"/> updates.
/// Concurrent requests are safe; on CUDA the model serializes their native work step by step (P-016), on CPU they run in
/// parallel.
/// </para>
/// <para>
/// <b>Cancellation.</b> A token cancelled before the call starts no work. Afterwards it is checked between prefill chunks
/// and between decode steps: the running chunk or step finishes, then the request ends with
/// <see cref="OperationCanceledException"/>, its native generator is released, and the model stays usable. Stopping a
/// streaming enumeration early (break or dispose) cancels the worker the same way, and disposal waits for it to exit.
/// </para>
/// <para>
/// <b>Lifetime.</b> An instance from <see cref="Create"/> owns its model; one wrapping an existing
/// <see cref="Qwen35ChatModel"/> owns nothing. <see cref="Dispose"/> is idempotent; later calls fail with
/// <see cref="ObjectDisposedException"/>. Requests already running keep the native model alive until they finish.
/// </para>
/// </remarks>
public sealed class Qwen35ChatClient : IChatClient
{
    /// <summary>Reported as <see cref="ChatClientMetadata.ProviderName"/>.</summary>
    public const string ProviderName = "starlings.onnxruntime-genai";

    /// <summary>
    /// Key in <see cref="ChatOptions.AdditionalProperties"/> that sets the chat template's <c>enable_thinking</c> for one
    /// request (a <see cref="bool"/>, or a JSON <c>true</c>/<c>false</c>), overriding
    /// <see cref="Qwen35ChatOptions.EnableThinking"/> of <see cref="Qwen35ChatClientOptions.Defaults"/>.
    /// </summary>
    public const string EnableThinkingPropertyName = "enable_thinking";

    private readonly Qwen35ChatModel _model;
    private readonly bool _ownsModel;
    private readonly Qwen35ChatClientOptions _options;
    private readonly ChatClientMetadata _metadata;
    private int _disposed;

    /// <summary>Wraps an existing model, which is neither owned nor disposed by this instance.</summary>
    public Qwen35ChatClient(Qwen35ChatModel model, Qwen35ChatClientOptions? options = null)
        : this(model, options ?? new Qwen35ChatClientOptions(), ownsModel: false)
    {
    }

    private Qwen35ChatClient(Qwen35ChatModel model, Qwen35ChatClientOptions options, bool ownsModel)
    {
        ArgumentNullException.ThrowIfNull(model);
        options.Validate();
        _model = model;
        _options = options;
        _ownsModel = ownsModel;
        ModelId = model.Info.Checkpoint;
        _metadata = new ChatClientMetadata(ProviderName, providerUri: null, ModelId);
    }

    /// <summary>
    /// The model id: repository and full commit SHA of the checkpoint (<c>Qwen/Qwen3.5-0.8B@2fc06364...</c>), reported as
    /// <see cref="ChatClientMetadata.DefaultModelId"/> and <see cref="ChatResponse.ModelId"/>, and the only value accepted
    /// in <see cref="ChatOptions.ModelId"/>. The variant and provider are in <see cref="Qwen35ModelInfo"/>.
    /// </summary>
    public string ModelId { get; }

    /// <summary>The underlying model.</summary>
    public Qwen35ChatModel Model => _model;

    /// <summary>Options in effect.</summary>
    public Qwen35ChatClientOptions Options => _options;

    /// <summary>Provider name, model id, as <see cref="GetService"/> returns it.</summary>
    public ChatClientMetadata Metadata => _metadata;

    /// <summary>Loads a verified variant (<see cref="Qwen35ChatModel.Load"/>; never downloads) and owns the model.</summary>
    public static Qwen35ChatClient Create(Qwen35ModelFiles files, Qwen35ModelOptions? modelOptions = null, Qwen35ChatClientOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(files);
        options ??= new Qwen35ChatClientOptions();
        options.Validate();
        var model = Qwen35ChatModel.Load(files, modelOptions);
        try
        {
            return new Qwen35ChatClient(model, options, ownsModel: true);
        }
        catch
        {
            model.Dispose();
            throw;
        }
    }

    /// <inheritdoc />
    /// <remarks>Runs on a dedicated thread; see the class remarks for the mapping, cancellation and threads.</remarks>
    public async Task<ChatResponse> GetResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null, CancellationToken cancellationToken = default)
    {
        var (conversation, chatOptions) = Prepare(messages, options);
        cancellationToken.ThrowIfCancellationRequested();
        GenerationResult result = await RunOnDedicatedThread(() => _model.Generate(conversation, chatOptions, cancellationToken)).ConfigureAwait(false);
        string id = NewId();
        var message = new ChatMessage(ChatRole.Assistant, result.Text.Length > 0 ? [new TextContent(result.Text)] : [])
        {
            MessageId = id,
        };
        return new ChatResponse(message)
        {
            ResponseId = id,
            ModelId = ModelId,
            CreatedAt = DateTimeOffset.UtcNow,
            FinishReason = MapFinishReason(result.FinishReason),
            Usage = MapUsage(result.Usage),
            RawRepresentation = result,
        };
    }

    /// <inheritdoc />
    /// <remarks>
    /// Validation errors surface on the first <c>MoveNextAsync</c>. Generation runs on a dedicated thread; see the class
    /// remarks for the update contract and cancellation.
    /// </remarks>
    public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
        IEnumerable<ChatMessage> messages,
        ChatOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var (conversation, chatOptions) = Prepare(messages, options);
        cancellationToken.ThrowIfCancellationRequested();
        var channel = Channel.CreateBounded<GenerationUpdate>(new BoundedChannelOptions(_options.StreamingBufferCapacity)
        {
            SingleReader = true,
            SingleWriter = true,
            FullMode = BoundedChannelFullMode.Wait,
        });
        using var stop = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        CancellationToken workerToken = stop.Token;
        Task worker = RunOnDedicatedThread(() =>
        {
            try
            {
                foreach (GenerationUpdate update in _model.GenerateStreaming(conversation, chatOptions, workerToken))
                {
                    while (!channel.Writer.TryWrite(update))
                    {
                        // A dedicated thread: blocking here costs no thread-pool thread.
                        if (!channel.Writer.WaitToWriteAsync(workerToken).AsTask().GetAwaiter().GetResult())
                        {
                            return false;
                        }
                    }
                }

                return true;
            }
            finally
            {
                channel.Writer.TryComplete();
            }
        });

        string id = NewId();
        var createdAt = DateTimeOffset.UtcNow;
        bool finished = false;
        try
        {
            while (await channel.Reader.WaitToReadAsync(CancellationToken.None).ConfigureAwait(false))
            {
                while (channel.Reader.TryRead(out GenerationUpdate? update))
                {
                    var contents = new List<AIContent>(2);
                    if (update.Text.Length > 0)
                    {
                        contents.Add(new TextContent(update.Text));
                    }

                    var chatUpdate = new ChatResponseUpdate(ChatRole.Assistant, contents)
                    {
                        ResponseId = id,
                        MessageId = id,
                        CreatedAt = createdAt,
                        ModelId = ModelId,
                        RawRepresentation = update,
                    };
                    if (update.Result is { } result)
                    {
                        chatUpdate.FinishReason = MapFinishReason(result.FinishReason);
                        contents.Add(new UsageContent(MapUsage(result.Usage)));
                    }

                    yield return chatUpdate;
                }
            }

            // Rethrows the worker's OperationCanceledException or failure, if any.
            await worker.ConfigureAwait(false);
            finished = true;
        }
        finally
        {
            if (!finished)
            {
                // The consumer stopped early, or the worker failed: stop the worker at its next check and wait for it, so
                // the native generator is released before the enumerator's disposal completes.
                await stop.CancelAsync().ConfigureAwait(false);
                try
                {
                    await worker.ConfigureAwait(false);
                }
#pragma warning disable CA1031 // Already reported to the consumer, or irrelevant after an early stop.
                catch (Exception)
#pragma warning restore CA1031
                {
                }
            }
        }
    }

    /// <inheritdoc />
    public object? GetService(Type serviceType, object? serviceKey = null)
    {
        ArgumentNullException.ThrowIfNull(serviceType);
        if (serviceKey is not null)
        {
            return null;
        }

        if (serviceType == typeof(ChatClientMetadata))
        {
            return _metadata;
        }

        if (serviceType.IsInstanceOfType(this))
        {
            return this;
        }

        if (serviceType == typeof(Qwen35ChatModel))
        {
            return _model;
        }

        return serviceType == typeof(Qwen35ModelInfo) ? _model.Info : null;
    }

    /// <summary>Disposes the model if this instance owns it. Idempotent.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0 && _ownsModel)
        {
            _model.Dispose();
        }
    }

    internal static ChatFinishReason MapFinishReason(GenerationFinishReason reason) => reason switch
    {
        GenerationFinishReason.Stop or GenerationFinishReason.StopSequence => ChatFinishReason.Stop,
        GenerationFinishReason.Length => ChatFinishReason.Length,
        _ => throw new ArgumentOutOfRangeException(nameof(reason), reason, "Unknown finish reason."),
    };

    private static UsageDetails MapUsage(GenerationUsage usage) => new()
    {
        InputTokenCount = usage.PromptTokens,
        OutputTokenCount = usage.CompletionTokens,
        TotalTokenCount = usage.TotalTokens,
    };

    private static string NewId() => "qwen35-" + Guid.NewGuid().ToString("N");

    // Runs blocking native work on a new background thread and completes the task with its result, its failure, or as
    // cancelled for an OperationCanceledException.
    private static Task<T> RunOnDedicatedThread<T>(Func<T> work)
    {
        var completion = new TaskCompletionSource<T>(TaskCreationOptions.RunContinuationsAsynchronously);
        var thread = new Thread(() =>
        {
            try
            {
                completion.TrySetResult(work());
            }
            catch (OperationCanceledException ex)
            {
                completion.TrySetCanceled(ex.CancellationToken);
            }
#pragma warning disable CA1031 // Every failure is delivered through the task.
            catch (Exception ex)
#pragma warning restore CA1031
            {
                completion.TrySetException(ex);
            }
        })
        {
            IsBackground = true,
            Name = "Qwen35ChatClient request",
        };
        thread.Start();
        return completion.Task;
    }

    private (List<Qwen35Message> Conversation, Qwen35ChatOptions Options) Prepare(IEnumerable<ChatMessage> messages, ChatOptions? options)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        var conversation = Qwen35ChatRequestMapper.MapMessages(messages, options?.Instructions);
        var chatOptions = Qwen35ChatRequestMapper.MapOptions(options, _options.Defaults, ModelId);
        return (conversation, chatOptions);
    }
}
