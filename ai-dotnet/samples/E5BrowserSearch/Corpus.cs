namespace E5BrowserSearch;

/// <summary>A small multilingual corpus written for this sample: travel, food, science and everyday questions.</summary>
public static class Corpus
{
    public static IReadOnlyList<(string Language, string Text)> Passages { get; } =
    [
        ("en", "The night train from Vienna reaches Venice early in the morning, in time for breakfast by the canals."),
        ("en", "To keep basil fresh, trim the stems and stand it in a glass of water on the kitchen counter, not in the fridge."),
        ("en", "Photosynthesis turns sunlight, water and carbon dioxide into sugar and oxygen inside the leaves of plants."),
        ("en", "If your laptop battery drains quickly, lower the screen brightness and close apps running in the background."),
        ("de", "Die Bibliothek ist am Samstag von zehn bis sechzehn Uhr geöffnet, sonntags bleibt sie geschlossen."),
        ("de", "Für einen guten Hefeteig sollte die Milch lauwarm sein, sonst geht der Teig nicht richtig auf."),
        ("de", "Der Rhein ist einer der längsten Flüsse Europas und fließt von den Alpen bis zur Nordsee."),
        ("fr", "La tour Eiffel a été construite pour l'Exposition universelle de 1889 à Paris."),
        ("fr", "Pour réussir une mayonnaise, tous les ingrédients doivent être à la même température."),
        ("fr", "Le vaccin aide le système immunitaire à reconnaître un virus avant de le rencontrer."),
        ("es", "El tren de alta velocidad une Madrid y Barcelona en unas dos horas y media."),
        ("es", "La paella valenciana tradicional lleva arroz, pollo, conejo, judías verdes y azafrán."),
        ("es", "Los terremotos se producen cuando las placas tectónicas liberan energía de forma repentina."),
        ("it", "Il Colosseo poteva ospitare decine di migliaia di spettatori per gli spettacoli dei gladiatori."),
        ("it", "Per una carbonara autentica si usano guanciale, pecorino, uova e pepe nero, senza panna."),
        ("pt", "O Rio Amazonas é o maior do mundo em volume de água e atravessa vários países da América do Sul."),
        ("nl", "Fietsen is in Nederland vaak sneller dan de auto voor korte afstanden in de stad."),
        ("sv", "Under midsommar dansar många svenskar runt midsommarstången och äter sill och jordgubbar."),
        ("pl", "Kopernik zaproponował model, w którym to Ziemia krąży wokół Słońca."),
        ("ja", "富士山は日本で最も高い山で、晴れた日には東京からも見えることがあります。"),
        ("ja", "おいしいご飯を炊くには、お米を研いでから三十分ほど水に浸しておきます。"),
        ("zh", "长城是中国古代修建的防御工程，全长超过两万公里。"),
        ("zh", "绿茶不宜用沸水冲泡，水温在八十度左右最好。"),
        ("ko", "서울의 지하철은 깨끗하고 빠르며 대부분의 관광지와 연결되어 있습니다."),
        ("ar", "تعد القهوة العربية رمزاً للضيافة وتقدم عادة مع التمر."),
        ("hi", "योग शरीर और मन दोनों को स्वस्थ रखने में मदद करता है।"),
        ("ru", "Транссибирская магистраль соединяет Москву с Владивостоком и считается самой длинной железной дорогой в мире."),
        ("tr", "İstanbul, Avrupa ve Asya kıtalarını birbirine bağlayan tarihi bir şehirdir."),
    ];

    /// <summary>
    /// Example queries, most in another language than their answer. Each ranks the passage it is about first (checked
    /// with browser/conformance/smoke_sample.py); e5-small ranks vaguer phrasings less reliably on a corpus this small.
    /// </summary>
    public static IReadOnlyList<string> ExampleQueries { get; } =
    [
        "how long should I soak rice before cooking",
        "Wann kann ich Bücher ausleihen?",
        "comment fonctionne la photosynthèse",
        "train from Madrid to Barcelona",
        "pasta without cream",
        "earthquakes and tectonic plates",
        "mountain visible from Tokyo",
    ];
}
