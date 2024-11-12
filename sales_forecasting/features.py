import pandas as pd


def col_name(prefix: str, iter: list[int]):
    return [f"{prefix}_{i}" for i in iter]


def build_shops_features(df: pd.DataFrame) -> pd.DataFrame:
    translation_map = {
        "!Якутск Орджоникидзе, 56 фран": "Yakutsk Ordzhonikidze, 56 Franchise",
        '!Якутск ТЦ "Центральный" фран': "Yakutsk Central Mall Franchise",
        'Адыгея ТЦ "Мега"': "Adygea Mega Mall",
        'Балашиха ТРК "Октябрь-Киномир"': "Balashikha October-Kinomir Mall",
        'Волжский ТЦ "Волга Молл"': "Volzhsky Volga Mall",
        'Вологда ТРЦ "Мармелад"': "Vologda Marmalade Mall",
        "Воронеж (Плехановская, 13)": "Voronezh Plekhanovskaya 13",
        'Воронеж ТРЦ "Максимир"': "Voronezh Maximir Mall",
        'Воронеж ТРЦ Сити-Парк "Град"': "Voronezh City Park Grad Mall",
        "Выездная Торговля": "Field Sales",
        "Жуковский ул. Чкалова 39м?": "Zhukovsky Chkalov St 39m?",
        "Жуковский ул. Чкалова 39м²": "Zhukovsky Chkalov St 39m²",
        "Интернет-магазин ЧС": "Internet Shop CH",
        'Казань ТЦ "Бехетле"': "Kazan Bekhetle Mall",
        'Казань ТЦ "ПаркХаус" II': "Kazan ParkHouse Mall II",
        'Калуга ТРЦ "XXI век"': "Kaluga 21st Century Mall",
        'Коломна ТЦ "Рио"': "Kolomna Rio Mall",
        'Красноярск ТЦ "Взлетка Плаза"': "Krasnoyarsk Vzletka Plaza Mall",
        'Красноярск ТЦ "Июнь"': "Krasnoyarsk June Mall",
        'Курск ТЦ "Пушкинский"': "Kursk Pushkinsky Mall",
        'Москва "Распродажа"': "Moscow Sale",
        'Москва МТРЦ "Афи Молл"': "Moscow AfiMall City",
        "Москва Магазин С21": "Moscow Shop S21",
        'Москва ТК "Буденовский" (пав.А2)': "Moscow Budenovsky Mall (A2)",
        'Москва ТК "Буденовский" (пав.К7)': "Moscow Budenovsky Mall (K7)",
        'Москва ТРК "Атриум"': "Moscow Atrium Mall",
        'Москва ТЦ "Ареал" (Беляево)': "Moscow Areal Mall (Belyaevo)",
        'Москва ТЦ "МЕГА Белая Дача II"': "Moscow Mega White Cottage II",
        'Москва ТЦ "МЕГА Теплый Стан" II': "Moscow Mega Teply Stan II",
        'Москва ТЦ "Новый век" (Новокосино)': "Moscow New Century Mall (Novokosino)",
        'Москва ТЦ "Перловский"': "Moscow Perlovo Mall",
        'Москва ТЦ "Семеновский"': "Moscow Semyonovsky Mall",
        'Москва ТЦ "Серебряный Дом"': "Moscow Silver House Mall",
        'Мытищи ТРК "XL-3"': "Mytishchi XL-3 Mall",
        'Н.Новгород ТРЦ "РИО"': "Nizhny Novgorod Rio Mall",
        'Н.Новгород ТРЦ "Фантастика"': "Nizhny Novgorod Fantastika Mall",
        'Новосибирск ТРЦ "Галерея Новосибирск"': "Novosibirsk Gallery Mall",
        'Новосибирск ТЦ "Мега"': "Novosibirsk Mega Mall",
        'Омск ТЦ "Мега"': "Omsk Mega Mall",
        'РостовНаДону ТРК "Мегацентр Горизонт"': "RostovOnDon Horizon MegaCenter",
        'РостовНаДону ТРК "Мегацентр Горизонт" Островной': "RostovOnDon Horizon MegaCenter Island",
        'РостовНаДону ТЦ "Мега"': "RostovOnDon Mega Mall",
        'СПб ТК "Невский Центр"': "St.Petersburg Nevsky Center Mall",
        'СПб ТК "Сенная"': "St.Petersburg Sennaya Mall",
        'Самара ТЦ "Мелодия"': "Samara Melody Mall",
        'Самара ТЦ "ПаркХаус"': "Samara ParkHouse Mall",
        'Сергиев Посад ТЦ "7Я"': "Sergiev Posad 7Ya Mall",
        'Сургут ТРЦ "Сити Молл"': "Surgut City Mall",
        'Томск ТРЦ "Изумрудный Город"': "Tomsk Emerald City Mall",
        'Тюмень ТРЦ "Кристалл"': "Tyumen Crystal Mall",
        'Тюмень ТЦ "Гудвин"': "Tyumen Goodwin Mall",
        'Тюмень ТЦ "Зеленый Берег"': "Tyumen Green Shore Mall",
        'Уфа ТК "Центральный"': "Ufa Central Mall",
        'Уфа ТЦ "Семья" 2': "Ufa Family Mall 2",
        'Химки ТЦ "Мега"': "Khimki Mega Mall",
        "Цифровой склад 1С-Онлайн": "Digital Warehouse 1C-Online",
        'Чехов ТРЦ "Карнавал"': "Chekhov Carnival Mall",
        "Якутск Орджоникидзе, 56": "Yakutsk Ordzhonikidze, 56",
        'Якутск ТЦ "Центральный"': "Yakutsk Central Mall",
        'Ярославль ТЦ "Альтаир"': "Yaroslavl Altair Mall",
    }

    df["shop_name_en"] = df["shop_name"].map(translation_map)
    df[["city", "shop_name_direct"]] = df["shop_name_en"].str.split(n=1, expand=True).values
    df["city_id"] = pd.factorize(df["city"])[0]

    return df


def build_item_categories_features(df: pd.DataFrame) -> pd.DataFrame:
    """This df has been translated by GPT"""
    df = pd.read_parquet(".data/item_categories_features.parquet")
    df[["general_item_category_name", "specific_item_category_name"]] = df["item_category_name_en"].str.split(r"\s-\s", expand=True).values
    df["general_item_category_id"] = pd.factorize(df["general_item_category_name"])[0]
    df = df.fillna("[Empty]")
    return df


def build_items_features(df: pd.DataFrame) -> pd.DataFrame:
    from sklearn.feature_extraction.text import TfidfVectorizer

    n_feats = 1000
    tfidf = TfidfVectorizer(max_features=n_feats)
    tfidf_features = tfidf.fit_transform(df["item_name"])
    tfidf_features_array = tfidf_features.toarray()

    for i in range(n_feats):
        df[f"item_name_tfidf_{i}"] = tfidf_features_array[:, i]

    return df


def build_month_map(sales_train: pd.DataFrame) -> dict[int, int]:
    """Builds mapping from `date_block_num` to month-in-year index between 0 and 11"""
    sales_train["date_month"] = pd.to_datetime(sales_train["date"], format="%d.%m.%Y").dt.month - 1

    return sales_train[["date_block_num", "date_month"]].drop_duplicates().set_index("date_block_num")["date_month"].to_dict()


def build_train_df(
    sales_train: pd.DataFrame, shops: pd.DataFrame, items: pd.DataFrame, item_categories: pd.DataFrame, month_map: dict[int, int]
) -> pd.DataFrame:
    """Build train df by joining the features."""
    df_train = (
        sales_train.merge(shops, on="shop_id", how="left")
        .merge(items, on="item_id", how="left")
        .merge(item_categories, on="item_category_id")
        .assign(date_month=sales_train["date_block_num"].map(month_map))
        .sort_values(by=["shop_id", "item_id", "date_block_num"])
        .reset_index(drop=True)
    )
    return df_train


def build_test_df(
    sales_train: pd.DataFrame,
    test: pd.DataFrame,
    shops: pd.DataFrame,
    items: pd.DataFrame,
    item_categories: pd.DataFrame,
    month_map: dict[int, int],
) -> pd.DataFrame:
    """Build test df by joining the features."""
    max_month = sales_train["date_block_num"].max()

    df_test = (
        test.merge(shops, on="shop_id")
        .merge(items, on="item_id")
        .merge(item_categories, on="item_category_id")
        .assign(date_block_num=max_month + 1, date_month=month_map[max_month] + 1)
        .drop(columns=["ID"])
        .sort_values(by=["shop_id", "item_id", "date_block_num"])
        .reset_index(drop=True)
    )
    return df_test
