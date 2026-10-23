// Multilingual semantic search with intfloat/multilingual-e5-small, entirely in the browser.
//
//   dotnet run                         dev server (one inference thread: it sends no cross-origin isolation headers)
//   dotnet publish -c Release          static files for any host; send COOP/COEP for threads (README)
using E5BrowserSearch;
using Microsoft.AspNetCore.Components.WebAssembly.Hosting;

// The app runs only in the browser; the Starlings browser APIs are marked [SupportedOSPlatform("browser")].
[assembly: System.Runtime.Versioning.SupportedOSPlatform("browser")]

var builder = WebAssemblyHostBuilder.CreateDefault(args);
builder.RootComponents.Add<App>("#app");
builder.Services.AddSingleton<SearchEngine>();
await builder.Build().RunAsync();
