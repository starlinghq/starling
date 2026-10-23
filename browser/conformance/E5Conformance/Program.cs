using Microsoft.AspNetCore.Components.WebAssembly.Hosting;
using Starlings.E5Conformance;

// The app runs only in the browser; the Starlings browser APIs are marked [SupportedOSPlatform("browser")].
[assembly: System.Runtime.Versioning.SupportedOSPlatform("browser")]

var builder = WebAssemblyHostBuilder.CreateDefault(args);
builder.RootComponents.Add<App>("#app");
ConformanceEntryPoints.Initialize(new Uri(builder.HostEnvironment.BaseAddress));
await builder.Build().RunAsync();
