using Microsoft.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;

namespace Starlings.Models.Qwen35.Extensions.AI;

/// <summary>Registers a <see cref="Qwen35ChatClient"/> in a service collection.</summary>
public static class Qwen35ChatClientServiceCollectionExtensions
{
    /// <summary>
    /// Registers one <see cref="Qwen35ChatClient"/> singleton, created by <see cref="Qwen35ChatClient.Create"/> from
    /// <paramref name="files"/> on first resolution, and <see cref="IChatClient"/> resolving to the same instance unless one
    /// is already registered.
    /// </summary>
    /// <remarks>
    /// The container owns the client, which owns the model: disposing the service provider releases it. To add
    /// Microsoft.Extensions.AI middleware, register the pipeline after this call, for example
    /// <c>services.AddChatClient(sp =&gt; sp.GetRequiredService&lt;Qwen35ChatClient&gt;()).UseLogging()</c>; the later
    /// registration wins, and disposing the pipeline disposes the client, which is idempotent.
    /// </remarks>
    public static IServiceCollection AddQwen35ChatClient(
        this IServiceCollection services, Qwen35ModelFiles files, Qwen35ModelOptions? modelOptions = null, Qwen35ChatClientOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(files);
        (options ?? new Qwen35ChatClientOptions()).Validate();
        return services.AddQwen35ChatClient(_ => Qwen35ChatClient.Create(files, modelOptions, options));
    }

    /// <summary>
    /// Registers one <see cref="Qwen35ChatClient"/> singleton from <paramref name="factory"/> and <see cref="IChatClient"/>
    /// resolving to the same instance unless one is already registered. The container disposes the instance the factory
    /// returns.
    /// </summary>
    public static IServiceCollection AddQwen35ChatClient(this IServiceCollection services, Func<IServiceProvider, Qwen35ChatClient> factory)
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(factory);
        services.AddSingleton(factory);
        services.TryAddSingleton<IChatClient>(sp => sp.GetRequiredService<Qwen35ChatClient>());
        return services;
    }
}
