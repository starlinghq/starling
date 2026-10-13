using Starlings.BrowserSpike;
using Microsoft.AspNetCore.Components.WebAssembly.Hosting;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.JSInterop;

var builder = WebAssemblyHostBuilder.CreateDefault(args);
builder.RootComponents.Add<App>("#app");

WebAssemblyHost host = builder.Build();
SpikeEntryPoints.Initialize(new Uri(builder.HostEnvironment.BaseAddress), host.Services.GetRequiredService<IJSRuntime>());
await host.RunAsync();
