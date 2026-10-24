// The bundle has no UI and nothing to start: JavaScript calls E5Exports (Exports.cs) after dotnet.create().
// The APIs it uses are browser-only ([SupportedOSPlatform("browser")]).
[assembly: System.Runtime.Versioning.SupportedOSPlatform("browser")]

return 0;
