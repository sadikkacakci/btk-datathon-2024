import pandas as pd
import warnings
import numpy as np
import re
warnings.filterwarnings("ignore")

class TrainDataProcessor:
    def __init__(self,file_path):
        self.raw_data = pd.read_csv(file_path)
        self.il_ilce_path = "data/il_ilce.csv"

    def drop_high_nan_columns(self,df,threshold_nan_ratio=30):
        nan_ratios = df.isna().mean() * 100
        # NaN oranı %50'den fazla olan sütunları filtrele
        columns_with_high_nan = nan_ratios[nan_ratios > threshold_nan_ratio].index.tolist()
        columns_with_high_nan.append("id")
        columns_with_high_nan.append("Ikametgah Sehri")
        return columns_with_high_nan

    def fix_cinsiyet_column(self,df):
        df['Cinsiyet'] = df['Cinsiyet'].replace('ERKEK', 'Erkek')
        df = df.loc[df['Cinsiyet'] != 'Belirtmek istemiyorum']
        df['Cinsiyet'] = df['Cinsiyet'].replace('Erkek', 1)
        df['Cinsiyet'] = df['Cinsiyet'].replace('Kadın', 0)
        return df
    
    def fix_dogum_tarihi(self,df):
        pattern_2000s_short = r'\b\d{1,2}/\d{1,2}/0[0-9]\b'

        pattern_90s_short = r'\b9\d\b'

        pattern_1970s = r'\b197\d\b'
        pattern_1980s = r'\b198\d\b'
        pattern_1990s = r'\b199\d\b'
        pattern_2000s = r'\b200\d\b'
        pattern_2010s = r'\b201\d\b'

        def extract_and_replace_year(date_str):
            if pd.isna(date_str):
                return np.nan
            match = re.search(pattern_1970s, date_str)
            if match:
                return match.group(0)
            match = re.search(pattern_1980s, date_str)
            if match:
                return match.group(0)
            match = re.search(pattern_1990s, date_str)
            if match:
                return match.group(0)
            match = re.search(pattern_2000s, date_str)
            if match:
                return match.group(0)
            match = re.search(pattern_2010s, date_str)
            if match:
                return match.group(0)
            # 2 haneli 90'lar (90-99)
            match = re.search(pattern_90s_short, date_str)
            if match:
                return '19' + match.group(0)  # Yılın başına '19' ekleyerek 1990'lı yılları elde ediyoruz
            
            # 2 haneli 2000'ler (01-09)
            match = re.search(pattern_2000s_short, date_str)
            if match:
                year_part = match.group(0)[-2:]  # Yılın son iki hanesini alıyoruz
                return '20' + year_part  # Başına 20 ekliyoruz, örneğin 01 -> 2001
            
            return np.nan  # Yıl bulunamazsa nan döndürür, buradan veri kaybı çok. Düzeltilebilir.
        df['Dogum Tarihi'] = df['Dogum Tarihi'].apply(extract_and_replace_year)
        df = df[df['Dogum Tarihi'] != '1/1/70 2:00']
        df = df.dropna(subset=['Dogum Tarihi'])
        return df

    def fix_dogum_yeri_column(self,df):
        def custom_lowercase(text):
            if text == "Iğdır":
                return "ığdır"
            if pd.isna(text):
                return text
            result = ""
            for char in text:
                # Büyük harflerden küçük harflere manuel dönüştürme
                if 'A' <= char <= 'Z':
                    result += chr(ord(char) + 32)
                elif 'Ç' == char:
                    result += 'ç'
                elif 'Ğ' == char:
                    result += 'ğ'
                elif 'İ' == char:
                    result += 'i'
                elif 'Ö' == char:
                    result += 'ö'
                elif 'Ş' == char:
                    result += 'ş'
                elif 'Ü' == char:
                    result += 'ü'
                else:
                    result += char  # Küçük harf veya başka karakterse olduğu gibi ekle
            return result
        
        def turkish_to_english(s):
            turkish_chars = "çğıöşüÇĞİÖŞÜ"
            english_chars = "cgiosuCGIOSU"
            
            translation_table = str.maketrans(turkish_chars, english_chars)
            return s.translate(translation_table)
        
        df['Dogum Yeri'] = df['Dogum Yeri'].apply(custom_lowercase)
        il_ilce_file_path = self.il_ilce_path
        il_ilce_df = pd.read_csv(il_ilce_file_path)
        il_ilce_df["il"] = il_ilce_df["il"].apply(custom_lowercase)
        il_ilce_df["ilce"] = il_ilce_df["ilce"].apply(custom_lowercase)
        # ilce_list = il_ilce_df["ilce"].tolist()
        il_list = il_ilce_df["il"].unique().tolist()
        il_list_english = [turkish_to_english(il) for il in il_list]
        # İl isimleri listesini ve İngilizce karşılıklarını oluştur
        il_list = il_ilce_df["il"].unique().tolist()
        il_list_english = [turkish_to_english(il) for il in il_list]
        # İl isimlerini ve İngilizce karşılıklarını içeren sözlüğü oluştur
        il_dict = dict(zip(il_list_english, il_list))
        for il in il_dict.keys():
            if df['Dogum Yeri'].str.contains(il).any():
                # city = il_ilce_df.loc[il_ilce_df["ilce"] == ilce,"il"].values[0]
                df.loc[df['Dogum Yeri'].str.contains(il, na=False), 'Dogum Yeri'] = il_dict.get(il)

        for il in il_list:
            if df['Dogum Yeri'].str.contains(il).any():
                df.loc[df['Dogum Yeri'].str.contains(il, na=False), 'Dogum Yeri'] = il

        # for ilce in ilce_list:
        #     if df['Dogum Yeri'].str.contains(ilce).any():
        #         city = il_ilce_df.loc[il_ilce_df["ilce"] == ilce,"il"].values[0]
        #         df.loc[df['Dogum Yeri'].str.contains(ilce, na=False), 'Dogum Yeri'] = city

        frequency = 20
        value_counts = df["Dogum Yeri"].value_counts()
        to_drop = value_counts[value_counts < frequency].index
        df = df[~df["Dogum Yeri"].isin(to_drop)]

        df["Dogum Yeri"] = df["Dogum Yeri"].replace("kadiköy","istanbul")
        df["Dogum Yeri"] = df["Dogum Yeri"].replace("kadıköy","istanbul")
        df["Dogum Yeri"] = df["Dogum Yeri"].replace("fatih","istanbul")
        df["Dogum Yeri"] = df["Dogum Yeri"].replace("üsküdar","istanbul")
        df["Dogum Yeri"] = df["Dogum Yeri"].replace("k.maraş","kahramanmaraş")
        df["Dogum Yeri"] = df["Dogum Yeri"].replace("afyon","afyonkarahisar")
        df["Dogum Yeri"] = df["Dogum Yeri"].replace("iğdır","ığdır")
        df["Dogum Yeri"] = df["Dogum Yeri"].replace("elaziğ","elazığ")
        df["Dogum Yeri"] = df["Dogum Yeri"].replace("altindağ","ankara")
        df["Dogum Yeri"] = df["Dogum Yeri"].replace("altındağ","ankara")
        df["Dogum Yeri"] = df["Dogum Yeri"].replace("bakirköy","istanbul")
        df["Dogum Yeri"] = df["Dogum Yeri"].replace("hakkâri","hakkari")
        df["Dogum Yeri"] = df["Dogum Yeri"].replace("antakya","hatay")
        df["Dogum Yeri"] = df["Dogum Yeri"].replace("gümüshane","gümüşhane")
        df["Dogum Yeri"] = df["Dogum Yeri"].replace("bakırköy","istanbul")
        df["Dogum Yeri"] = df["Dogum Yeri"].replace("konak","izmir")
        df["Dogum Yeri"] = df["Dogum Yeri"].replace("izmit","kocaeli")
        df["Dogum Yeri"] = df["Dogum Yeri"].replace("çankiri","çankırı")
        df["Dogum Yeri"] = df["Dogum Yeri"].replace("osmangazi","eskişehir")
        df["Dogum Yeri"] = df["Dogum Yeri"].replace("şişli","istanbul")
        df["Dogum Yeri"] = df["Dogum Yeri"].replace("yenimahalle","ankara")

        df["Dogum Yeri"] = df["Dogum Yeri"].replace("diger","kktc")
        df["Dogum Yeri"] = df["Dogum Yeri"].replace("diğer","yurt dışı")
        df = df[df['Dogum Yeri'] != '------']

        file_path = "data/SehirlerBolgeler.csv"
        sehir_bolge_df = pd.read_csv(file_path)
        sehir_bolge_df["SehirAd"] = sehir_bolge_df["SehirAd"].apply(custom_lowercase)
        sehir_list = sehir_bolge_df["SehirAd"].tolist()
        for sehir in sehir_list:
            if sehir == "istanbul" or sehir == "ankara":
                continue
            bolge = sehir_bolge_df.loc[sehir_bolge_df["SehirAd"] == sehir,"BolgeAd"].values[0]
            df.loc[df["Dogum Yeri"] == sehir,"Dogum Yeri"] = bolge
        df["Dogum Yeri"] = df["Dogum Yeri"].replace("yurt dışı","diger")
        df["Dogum Yeri"] = df["Dogum Yeri"].replace("kktc","diger")

        columns_to_convert = []
        unique_elements = df["Dogum Yeri"].unique()
        for unique in unique_elements:
            if pd.isna(unique):
                continue
            columns_to_convert.append(f"memleket_{unique}")
        df = pd.get_dummies(df, columns=["Dogum Yeri"], prefix="memleket")
        df[columns_to_convert] = df[columns_to_convert].astype(int)
        return df

    def fix_universite_turu_column(self,df):
        df["Universite Turu"] = df["Universite Turu"].replace('DEVLET', 'Devlet')
        df["Universite Turu"] = df["Universite Turu"].replace('ÖZEL', 'Özel')
        df['Universite Turu'] = df['Universite Turu'].replace('Özel', 1)
        df['Universite Turu'] = df['Universite Turu'].replace('Devlet', 0)
        return df

    def fix_burs_aliyor_mu_column(self,df):
        df["Burs Aliyor mu?"] = df["Burs Aliyor mu?"].replace('EVET', 'Evet')
        df["Burs Aliyor mu?"] = df["Burs Aliyor mu?"].replace('evet', 'Evet')
        df["Burs Aliyor mu?"] = df["Burs Aliyor mu?"].replace('hayır', 'Hayır')
        df['Burs Aliyor mu?'] = df['Burs Aliyor mu?'].replace('Evet', 1)
        df['Burs Aliyor mu?'] = df['Burs Aliyor mu?'].replace('Hayır', 0)
        return df

    def fix_universite_kacinci_sinif_column(self,df):
        # Universite Kacinci Sinif, burada bir ordinallik yok.
        df = df[~df["Universite Kacinci Sinif"].isin(["Mezun", "Yüksek Lisans", "Tez", "0"])]
        df["Universite Kacinci Sinif"] = df["Universite Kacinci Sinif"].replace("hazırlık","Hazırlık")

        columns_to_convert = []
        unique_elements = df["Universite Kacinci Sinif"].unique()
        for unique in unique_elements:
            if pd.isna(unique):
                continue
            columns_to_convert.append(f"is_{unique}")
        df = pd.get_dummies(df, columns=["Universite Kacinci Sinif"], prefix="is")
        df[columns_to_convert] = df[columns_to_convert].astype(int)
        return df

    # def fix_lise_turu_column(self,df,distribute_devlet_value=True):
    #     # Lise Turu
    #     df['Lise Turu'] = df['Lise Turu'].replace('Meslek lisesi', 'Meslek Lisesi')
    #     df['Lise Turu'] = df['Lise Turu'].replace('Meslek', 'Meslek Lisesi')
    #     df['Lise Turu'] = df['Lise Turu'].replace('Düz lise', 'Anadolu Lisesi')
    #     df['Lise Turu'] = df['Lise Turu'].replace('Anadolu lisesi', 'Anadolu Lisesi')
    #     df['Lise Turu'] = df['Lise Turu'].replace('Düz Lise', 'Anadolu Lisesi')
    #     df['Lise Turu'] = df['Lise Turu'].replace('Özel', 'Özel Lise')
    #     df['Lise Turu'] = df['Lise Turu'].replace('Özel lisesi', 'Özel Lise')
    #     df['Lise Turu'] = df['Lise Turu'].replace('Özel Lisesi', 'Özel Lise')
    #     df['Lise Turu'] = df['Lise Turu'].replace('Fen lisesi', 'Fen Lisesi')

    #     ''' Devlet değerinin dağıtılması -> Anadolu Lisesi, Meslek Lisesi, İmam Hatip Lisesi '''
    #     if distribute_devlet_value:
    #         # Mevcut dağılım oranlarını hesapla
    #         counts = df['Lise Turu'].value_counts()
    #         # Anadolu Lisesi, Meslek Lisesi ve İmam Hatip Lisesi için toplam frekans
    #         total = counts['Anadolu Lisesi'] + counts['Meslek Lisesi'] + counts['İmam Hatip Lisesi'] + counts['Fen Lisesi']
    #         # Oranları hesapla
    #         anadolu_ratio = counts['Anadolu Lisesi'] / total
    #         meslek_ratio = counts['Meslek Lisesi'] / total
    #         imamhatip_ratio = counts['İmam Hatip Lisesi'] / total
    #         fen_lisesi_ratio = counts['Fen Lisesi'] / total
    #         # "Devlet" olan değerlerin sayısını al
    #         devlet_count = counts['Devlet']
    #         # Devlet değerlerini oranlara göre dağıt
    #         anadolu_count = int(devlet_count * anadolu_ratio)
    #         meslek_count = int(devlet_count * meslek_ratio)
    #         imamhatip_count = int(devlet_count * imamhatip_ratio)
    #         fen_lisesi_count = devlet_count - (anadolu_count + meslek_count + imamhatip_count)  # Kalanını Fen Lisesi'ne ata
    #         # Yeni değerleri oluştur
    #         new_values = (['Anadolu Lisesi'] * anadolu_count +
    #                     ['Meslek Lisesi'] * meslek_count +
    #                     ['İmam Hatip Lisesi'] * imamhatip_count +
    #                     ['Fen Lisesi'] * fen_lisesi_count)

    #         # "Devlet" olan satırların indexlerini bul
    #         devlet_indices = df[df['Lise Turu'] == 'Devlet'].index

    #         # Yeni değerleri karıştır ve Devlet satırlarına dağıt
    #         np.random.shuffle(new_values)  # Karıştır
    #         df.loc[devlet_indices, 'Lise Turu'] = new_values
    #     return df

    def fix_lise_turu_column(self,df):
        df['Lise Turu'] = df['Lise Turu'].replace('Meslek Lisesi', 'Devlet')
        df['Lise Turu'] = df['Lise Turu'].replace('Meslek lisesi', 'Devlet')
        df['Lise Turu'] = df['Lise Turu'].replace('Meslek', 'Devlet')
        df['Lise Turu'] = df['Lise Turu'].replace('Düz lise', 'Devlet')
        df['Lise Turu'] = df['Lise Turu'].replace('Anadolu lisesi', 'Devlet')
        df['Lise Turu'] = df['Lise Turu'].replace('Anadolu Lisesi', 'Devlet')
        df['Lise Turu'] = df['Lise Turu'].replace('Düz Lise', 'Devlet')
        df['Lise Turu'] = df['Lise Turu'].replace('Diğer', 'Devlet')
        df['Lise Turu'] = df['Lise Turu'].replace('Fen lisesi', 'Devlet')
        df['Lise Turu'] = df['Lise Turu'].replace('Fen Lisesi', 'Devlet')
        df['Lise Turu'] = df['Lise Turu'].replace('İmam Hatip Lisesi', 'Devlet')
        df['Lise Turu'] = df['Lise Turu'].replace('Özel lisesi', 'Özel')
        df['Lise Turu'] = df['Lise Turu'].replace('Özel Lisesi', 'Özel')
        df['Lise Turu'] = df['Lise Turu'].replace('Özel Lise', 'Özel')

        df["Lise Turu"] = df['Lise Turu'].replace('Özel', 1)
        df["Lise Turu"] = df['Lise Turu'].replace('Devlet', 0)
        return df

    def fix_lise_bolumu_column(self,df):
        frequency = 50
        value_counts = df["Lise Bolumu"].value_counts()
        to_drop = value_counts[value_counts < frequency].index

        # Frekansı 5'ten küçük olan değerleri dataframe'den çıkar
        df = df[~df["Lise Bolumu"].isin(to_drop)]
        df["Lise Bolumu"] = df["Lise Bolumu"].replace("FEN SAYISAL BİLİMLERİ ALANI","Sayısal")
        df["Lise Bolumu"] = df["Lise Bolumu"].replace("fen bilimleri","Sayısal")
        df["Lise Bolumu"] = df["Lise Bolumu"].replace("Matematik-Fen","Sayısal")
        df["Lise Bolumu"] = df["Lise Bolumu"].replace("SAYISAL","Sayısal")
        df["Lise Bolumu"] = df["Lise Bolumu"].replace("Fen","Sayısal")
        df["Lise Bolumu"] = df["Lise Bolumu"].replace("MF","Sayısal")
        df["Lise Bolumu"] = df["Lise Bolumu"].replace("Fen Bilimleri","Sayısal")
        df["Lise Bolumu"] = df["Lise Bolumu"].replace("sayısal","Sayısal")

        df["Lise Bolumu"] = df["Lise Bolumu"].replace("EŞİT AĞIRLIK TÜRKÇE-MATEMATİK ALANI","Eşit Ağırlık")
        df["Lise Bolumu"] = df["Lise Bolumu"].replace("eşit ağırlık","Eşit Ağırlık")
        df["Lise Bolumu"] = df["Lise Bolumu"].replace("Eşit ağırlık","Eşit Ağırlık")
        df["Lise Bolumu"] = df["Lise Bolumu"].replace("TM","Eşit Ağırlık")
        df["Lise Bolumu"] = df["Lise Bolumu"].replace("tm","Eşit Ağırlık")
        df["Lise Bolumu"] = df["Lise Bolumu"].replace("Tm","Eşit Ağırlık")
        df["Lise Bolumu"] = df["Lise Bolumu"].replace("Türkçe-Matematik","Eşit Ağırlık")

        df["Lise Bolumu"] = df["Lise Bolumu"].replace("Yabancı Dil","Dil")
        df["Lise Bolumu"] = df["Lise Bolumu"].replace("DİL ALANI","Dil")

        df["Lise Bolumu"] = df["Lise Bolumu"].replace("EŞİT AĞIRLIK","Eşit Ağırlık")
        df["Lise Bolumu"] = df["Lise Bolumu"].replace("FEN BİLİMLERİ","Sayısal")
        df["Lise Bolumu"] = df["Lise Bolumu"].replace("fen","Sayısal")
        df["Lise Bolumu"] = df["Lise Bolumu"].replace("Bilişim Teknolojileri","Sayısal")
        df["Lise Bolumu"] = df["Lise Bolumu"].replace("Türkçe Matematik","Eşit Ağırlık")
        df["Lise Bolumu"] = df["Lise Bolumu"].replace("SOSYAL ALANI","Sözel")
        df = df.loc[df["Lise Bolumu"] != "Diğer"]
        columns_to_convert = []
        unique_elements = df["Lise Bolumu"].unique()
        for unique in unique_elements:
            if pd.isna(unique):
                continue
            columns_to_convert.append(f"lise_bolumu_{unique}")
        df = pd.get_dummies(df, columns=["Lise Bolumu"], prefix="lise_bolumu")
        df[columns_to_convert] = df[columns_to_convert].astype(int)
        return df

    def fix_lise_mezuniyet_notu_column(self,df):
        frequency = 13
        value_counts = df["Lise Mezuniyet Notu"].value_counts()
        to_drop = value_counts[value_counts < frequency].index
        df = df[~df["Lise Mezuniyet Notu"].isin(to_drop)]
        # 1.part
        df["Lise Mezuniyet Notu"] = df["Lise Mezuniyet Notu"].replace("4.00-3.50","75-100")
        df["Lise Mezuniyet Notu"] = df["Lise Mezuniyet Notu"].replace("75 - 100","75-100")
        df["Lise Mezuniyet Notu"] = df["Lise Mezuniyet Notu"].replace("3.50-3.00","75-87.5")
        df["Lise Mezuniyet Notu"] = df["Lise Mezuniyet Notu"].replace("3.50-3","75-87.5")
        df["Lise Mezuniyet Notu"] = df["Lise Mezuniyet Notu"].replace("3.00 - 4.00","75-100")

        df["Lise Mezuniyet Notu"] = df["Lise Mezuniyet Notu"].replace("3.00-2.50","50-75")
        df["Lise Mezuniyet Notu"] = df["Lise Mezuniyet Notu"].replace("50 - 74","50-75")
        df["Lise Mezuniyet Notu"] = df["Lise Mezuniyet Notu"].replace("50 - 75","50-75")

        df["Lise Mezuniyet Notu"] = df["Lise Mezuniyet Notu"].replace("2.50 ve altı","50-62.5")

        # # 2.part
        df["Lise Mezuniyet Notu"] = df["Lise Mezuniyet Notu"].replace("84-70","75-100")
        df["Lise Mezuniyet Notu"] = df["Lise Mezuniyet Notu"].replace("100-85","75-100")
        df["Lise Mezuniyet Notu"] = df["Lise Mezuniyet Notu"].replace("75-87.5","75-100")
        df["Lise Mezuniyet Notu"] = df["Lise Mezuniyet Notu"].replace("50-75","50-74")
        df["Lise Mezuniyet Notu"] = df["Lise Mezuniyet Notu"].replace("69-55","50-74")
        df["Lise Mezuniyet Notu"] = df["Lise Mezuniyet Notu"].replace("50-62.5","50-74")
        df["Lise Mezuniyet Notu"] = df["Lise Mezuniyet Notu"].replace("25 - 50","25-49")

        df = df[df['Lise Mezuniyet Notu'] != '54-45']

        # Convert
        df["Lise Mezuniyet Notu"] = df["Lise Mezuniyet Notu"].replace("25-49",0)
        df["Lise Mezuniyet Notu"] = df["Lise Mezuniyet Notu"].replace("50-74",1)
        df["Lise Mezuniyet Notu"] = df["Lise Mezuniyet Notu"].replace("75-100",2)
        return df

    def fix_baska_bir_kurumdan_burs_aliyor_mu_column(self,df):
        df["Baska Bir Kurumdan Burs Aliyor mu?"] = df["Baska Bir Kurumdan Burs Aliyor mu?"].replace("Hayır",0)
        df["Baska Bir Kurumdan Burs Aliyor mu?"] = df["Baska Bir Kurumdan Burs Aliyor mu?"].replace("Evet",1)
        return df

    def fix_egitim_durumu(self,df,column_name): # column_name should be "Anne Egitim Durumu" or "Baba Egitim Durumu"
        if "Anne" in column_name:
            column_type = "anne"
        if "Baba" in column_name:
            column_type = "baba"
        
        df[column_name] = df[column_name].replace('İlkokul Mezunu', 'İlkokul')
        df[column_name] = df[column_name].replace('İLKOKUL MEZUNU', 'İlkokul')

        df[column_name] = df[column_name].replace('ORTAOKUL MEZUNU', 'Ortaokul')
        df[column_name] = df[column_name].replace('Ortaokul Mezunu', 'Ortaokul')

        df[column_name] = df[column_name].replace('LİSE', 'Lise')
        df[column_name] = df[column_name].replace('lise', 'Lise')
        df[column_name] = df[column_name].replace('Lise Mezunu', 'Lise')

        df[column_name] = df[column_name].replace('ÜNİVERSİTE', 'Üniversite')
        df[column_name] = df[column_name].replace('Üniversite Mezunu', 'Üniversite')

        df[column_name] = df[column_name].replace('EĞİTİM YOK', 'Eğitimi yok')
        df[column_name] = df[column_name].replace('Eğitim Yok', 'Eğitimi yok')
        df[column_name] = df[column_name].replace('Yüksek Lisans / Doktora', 'Doktora')
        df[column_name] = df[column_name].replace('Yüksek Lisans / Doktara', 'Yüksek Lisans')
        df[column_name] = df[column_name].replace('YÜKSEK LİSANS', 'Yüksek Lisans')
        df[column_name] = df[column_name].replace('DOKTORA', 'Doktora')

        df[column_name] = df[column_name].replace('Doktora', 'Üniversite')
        df[column_name] = df[column_name].replace('Yüksek Lisans', 'Üniversite')
        df[column_name] = df[column_name].replace('İlkokul', 'Eğitimi yok')
        df[column_name] = df[column_name].replace('Ortaokul', 'Eğitimi yok')



        if "0" in df[column_name].unique():
            df[column_name] = df[column_name].replace('0', 'Eğitimi yok')

        columns_to_convert = []
        unique_elements = df[column_name].unique()
        for unique in unique_elements:
            if pd.isna(unique):
                continue
            columns_to_convert.append(f"{column_type}_{unique}")

        if "Anne" in column_name:
            df = pd.get_dummies(df, columns=[column_name], prefix=column_type)
        if "Baba" in column_name:
            df = pd.get_dummies(df, columns=[column_name], prefix=column_type)
        df[columns_to_convert] = df[columns_to_convert].astype(int)
        return df
    
    def fix_calisma_durumu_column(self, df,column_name): # test verisinde emekli olmadığı için emekliyi evete çevirdim. Bunun üzerine düşünülebilir.
        df[column_name] = df[column_name].replace("Emekli","Evet") 
        df[column_name] = df[column_name].replace("Evet",1)
        df[column_name] = df[column_name].replace("Hayır",0)
        return df

    def fix_sektor_column(self, df,column_name): # Anne Sektor ve/veya Baba Sektor droplanabilir.
        df[column_name] = df[column_name].replace("ÖZEL SEKTÖR","Özel Sektör")
        df[column_name] = df[column_name].replace("KAMU","Kamu")
        df[column_name] = df[column_name].replace("DİĞER","Diğer")
        df[column_name] = df[column_name].replace("-","Sektör Yok")
        df[column_name] = df[column_name].replace("0","Sektör Yok")

        # NaN değerlerin dağıtılması
        counts = df[column_name].value_counts()
        count_ozel_sektor = counts["Özel Sektör"]
        count_kamu = counts["Kamu"]
        count_sektor_yok = counts["Sektör Yok"]
        count_diger = counts["Diğer"]

        total = count_ozel_sektor + count_kamu + count_sektor_yok + count_diger

        ratio_ozel_sektor = count_ozel_sektor / total
        ratio_kamu = count_kamu / total
        ratio_sektor_yok = count_sektor_yok / total
        ratio_diger = count_diger / total

        # NaN olan satır sayısını bul
        nan_count = df[column_name].isna().sum()

        # NaN değerleri oranlara göre dağıtmak için kaç NaN satırının her sektöre atanacağını hesapla
        fill_count_ozel_sektor = int(nan_count * ratio_ozel_sektor)
        fill_count_kamu = int(nan_count * ratio_kamu)
        fill_count_sektor_yok = int(nan_count * ratio_sektor_yok)
        fill_count_diger = nan_count - (fill_count_ozel_sektor + fill_count_kamu + fill_count_sektor_yok)

        new_values = (["Özel Sektör"] * fill_count_ozel_sektor + ["Kamu"] * fill_count_kamu + ["Sektör Yok"] * fill_count_sektor_yok + ["Diğer"] * fill_count_diger)
        np.random.shuffle(new_values)  # Karıştır
        # NaN olan satırların index'lerini bul
        nan_indices = df[df[column_name].isna()].index.tolist()
        df.loc[nan_indices, column_name] = new_values


        # df[column_name] = df[column_name].replace("Kamu","Sektör Var")
        # df[column_name] = df[column_name].replace("Özel Sektör","Sektör Var")
        # df[column_name] = df[column_name].replace("Diğer","Sektör Var")
        # df[column_name] = df[column_name].replace("Sektör Yok",0)
        # df[column_name] = df[column_name].replace("Sektör Var",1)

        columns_to_convert = []
        unique_elements = df[column_name].unique()
        for unique in unique_elements:
            if pd.isna(unique):
                continue
            columns_to_convert.append(f"{column_name}_{unique}")
        df = pd.get_dummies(df, columns=[column_name], prefix=f"{column_name}")
        df[columns_to_convert] = df[columns_to_convert].astype(int)
        return df

    def fix_kardes_sayisi_column(self, df):
        df["Kardes Sayisi"] = df["Kardes Sayisi"].astype(float)
        return df

    def fix_girisimcilik_kulupleri_column(self, df):
        df["Girisimcilik Kulupleri Tarzi Bir Kulube Uye misiniz?"] = df["Girisimcilik Kulupleri Tarzi Bir Kulube Uye misiniz?"].replace("Hayır",0)
        df["Girisimcilik Kulupleri Tarzi Bir Kulube Uye misiniz?"] = df["Girisimcilik Kulupleri Tarzi Bir Kulube Uye misiniz?"].replace("Evet",1)
        return df

    def fix_profesyonel_spor_dali_column(self, df):
        df["Profesyonel Bir Spor Daliyla Mesgul musunuz?"] = df["Profesyonel Bir Spor Daliyla Mesgul musunuz?"].replace("Hayır",0)
        df["Profesyonel Bir Spor Daliyla Mesgul musunuz?"] = df["Profesyonel Bir Spor Daliyla Mesgul musunuz?"].replace("Evet",1)
        return df

    def fix_stk_projesi_column(self, df):
        df.loc[(df['Aktif olarak bir STK üyesi misiniz?'] == 'Hayır') & df['Stk Projesine Katildiniz Mi?'].isnull(), 'Stk Projesine Katildiniz Mi?'] = 'Hayır'

        counts = df["Stk Projesine Katildiniz Mi?"].value_counts(dropna=False)

        count_hayir = counts["Hayır"]
        count_evet = counts["Evet"]
        total = count_hayir + count_evet
        ratio_hayir = count_hayir / total
        ratio_evet = count_evet / total

        nan_count = df["Stk Projesine Katildiniz Mi?"].isna().sum()

        fill_count_hayir = int(nan_count * ratio_hayir)
        fill_count_evet = nan_count - fill_count_hayir

        new_values = (["Hayır"] * fill_count_hayir + ["Evet"] * fill_count_evet)
        np.random.shuffle(new_values)  # Karıştır
        nan_indices = df[df["Stk Projesine Katildiniz Mi?"].isna()].index.tolist()
        df.loc[nan_indices, "Stk Projesine Katildiniz Mi?"] = new_values

        df["Stk Projesine Katildiniz Mi?"] = df["Stk Projesine Katildiniz Mi?"].replace("Hayır",0)
        df["Stk Projesine Katildiniz Mi?"] = df["Stk Projesine Katildiniz Mi?"].replace("Evet",1)
        return df

    def fix_stk_uyesi_column(self, df):
        df["Aktif olarak bir STK üyesi misiniz?"] = df["Aktif olarak bir STK üyesi misiniz?"].replace("Hayır",0)
        df["Aktif olarak bir STK üyesi misiniz?"] = df["Aktif olarak bir STK üyesi misiniz?"].replace("Evet",1)
        return df

    def fix_girisimcilik_deneyim_column(self, df):
        df["Girisimcilikle Ilgili Deneyiminiz Var Mi?"] = df["Girisimcilikle Ilgili Deneyiminiz Var Mi?"].replace("Evet",1)
        df["Girisimcilikle Ilgili Deneyiminiz Var Mi?"] = df["Girisimcilikle Ilgili Deneyiminiz Var Mi?"].replace("Hayır",0)
        return df

    def fix_ingilizce_biliyor_musunuz_column(self, df):
        df["Ingilizce Biliyor musunuz?"] = df["Ingilizce Biliyor musunuz?"].replace("Evet",1)
        df["Ingilizce Biliyor musunuz?"] = df["Ingilizce Biliyor musunuz?"].replace("Hayır",0)
        return df
    
    def fix_universite_not_ortalamasi_column(self, df):
        df["Universite Not Ortalamasi"] = df["Universite Not Ortalamasi"].replace("4-3.5","3.50 - 4.00")
        df["Universite Not Ortalamasi"] = df["Universite Not Ortalamasi"].replace("3.50-3","3.00 - 3.49")
        df["Universite Not Ortalamasi"] = df["Universite Not Ortalamasi"].replace("3.00-2.50","2.50 - 2.99")
        df["Universite Not Ortalamasi"] = df["Universite Not Ortalamasi"].replace("2.50 -3.00","2.50 - 2.99")
        df["Universite Not Ortalamasi"] = df["Universite Not Ortalamasi"].replace("3.00 - 3.50","3.00 - 3.49")
        df["Universite Not Ortalamasi"] = df["Universite Not Ortalamasi"].replace("4.0-3.5","3.50 - 4.00")
        df["Universite Not Ortalamasi"] = df["Universite Not Ortalamasi"].replace("2.50 - 3.00","2.50 - 2.99")
        df["Universite Not Ortalamasi"] = df["Universite Not Ortalamasi"].replace("ORTALAMA BULUNMUYOR","Ortalama bulunmuyor")

        # 2.50 ve altı değerlerinin dağıtılması.
        ratios = df["Universite Not Ortalamasi"].value_counts()
        count_0_179 = ratios["0 - 1.79"]
        count_180_249 = ratios["1.80 - 2.49"]
        total = count_0_179 + count_180_249
        count_250_ve_alti = ratios["2.50 ve altı"]
        ratio_0_179 = count_0_179 / total
        ratio_180_249 = count_180_249 / total
        count_new_0_1_79 = int(count_250_ve_alti * ratio_0_179)
        count_new_1_80_2_49 = count_250_ve_alti - count_new_0_1_79 # 1 eksik değeri buraya verdim.
        new_values = (['0 - 1.79'] * count_new_0_1_79 + ['1.80 - 2.49'] * count_new_1_80_2_49)
        indices_2_50_ve_alti = df.loc[df['Universite Not Ortalamasi'] == '2.50 ve altı'].index.tolist()
        np.random.shuffle(new_values)  # Karıştır
        df.loc[indices_2_50_ve_alti, 'Universite Not Ortalamasi'] = new_values

        # 3.00 - 4.00 değerlerinin dağıtılması.
        ratios = df["Universite Not Ortalamasi"].value_counts()
        count_300_349 = ratios["3.00 - 3.49"]
        count_350_400 = ratios["3.50 - 4.00"]
        total = count_300_349 + count_350_400
        count_300_400 = ratios["3.00 - 4.00"]
        ratio_300_349 = count_300_349 / total
        ratio_350_400 = count_350_400 / total
        count_new_300_349 = int(count_300_400 * ratio_300_349) 
        count_new_350_400 = count_300_400 - count_new_300_349
        new_values = (['3.00 - 3.49'] * count_new_300_349 + ['3.50 - 4.00'] * count_new_350_400)
        indices_300_400 = df.loc[df['Universite Not Ortalamasi'] == '3.00 - 4.00'].index.tolist()
        np.random.shuffle(new_values)  # Karıştır
        df.loc[indices_300_400, 'Universite Not Ortalamasi'] = new_values

        # Ortalama bulunmuyor ve Hazırlığım değerlerinin dağıtılması.
        df["Universite Not Ortalamasi"] = df["Universite Not Ortalamasi"].replace("Ortalama bulunmuyor","Hazırlığım") # kısa olması için ikisini de aynı yaptım.
        df["Universite Not Ortalamasi"] = df["Universite Not Ortalamasi"].replace("Not ortalaması yok","Hazırlığım")
        counts = df["Universite Not Ortalamasi"].value_counts()
        count_0_179 = counts["0 - 1.79"]
        count_180_249 = counts["1.80 - 2.49"]
        count_250_299 = counts["2.50 - 2.99"]
        count_300_349 = counts["3.00 - 3.49"]
        count_350_400 = counts["3.50 - 4.00"]
        total = count_0_179 + count_180_249 + count_250_299 + count_300_349 + count_350_400
        ratio_0_179 = count_0_179 / total
        ratio_180_249 = count_180_249 / total
        ratio_250_299 = count_250_299 / total
        ratio_300_349 = count_300_349 / total
        ratio_350_400 = count_350_400 / total
        count_hazirligim = counts["Hazırlığım"]
        count_new_0_179 = int(count_hazirligim * ratio_0_179)
        count_new_180_249 = int(count_hazirligim * ratio_180_249)
        count_new_250_299 = int(count_hazirligim * ratio_250_299)
        count_new_300_349 = int(count_hazirligim * ratio_300_349)
        count_new_350_400 = count_hazirligim - count_new_0_179 - count_new_180_249 - count_new_250_299 - count_new_300_349
        new_values = (['0 - 1.79'] * count_new_0_179 + ['1.80 - 2.49'] * count_new_180_249 + ["2.50 - 2.99"] * count_new_250_299 + ["3.00 - 3.49"] * count_new_300_349 + ["3.50 - 4.00"] * count_new_350_400)
        np.random.shuffle(new_values)
        indices_hazirliğim = df.loc[df['Universite Not Ortalamasi'] == 'Hazırlığım'].index.tolist()
        df.loc[indices_hazirliğim, 'Universite Not Ortalamasi'] = new_values
        df = df.loc[df["Universite Not Ortalamasi"] != "2.00 - 2.50"]
        df = df.loc[df["Universite Not Ortalamasi"] != "1.00 - 2.50"]

        df["Universite Not Ortalamasi"] = df["Universite Not Ortalamasi"].replace("0 - 1.79",0)
        df["Universite Not Ortalamasi"] = df["Universite Not Ortalamasi"].replace("1.80 - 2.49",1)
        df["Universite Not Ortalamasi"] = df["Universite Not Ortalamasi"].replace("2.50 - 2.99",2)
        df["Universite Not Ortalamasi"] = df["Universite Not Ortalamasi"].replace("3.00 - 3.49",3)
        df["Universite Not Ortalamasi"] = df["Universite Not Ortalamasi"].replace("3.50 - 4.00",4)
        return df

    def fix_universite_adi_column(self,df):
        universite_score_file = "data/university_avg_degerlendirme_puani.csv"
        universite_score = pd.read_csv(universite_score_file)
        def format_university_name(university_name):
            if pd.isna(university_name):
                return university_name
            # Üniversite ismini title case yap
            formatted_name = university_name.title()
            # Türkçe karakterlerde hatalı dönüşüm olmaması için düzeltme
            formatted_name = formatted_name.replace("İ", "İ").replace("Ü", "Ü").replace("Ö", "Ö").replace("Ğ", "Ğ").replace("Ç", "Ç").replace("Ş", "Ş")
            return formatted_name
        df["Universite Adi"] = df["Universite Adi"].apply(format_university_name)
        universite_adi_list = df["Universite Adi"].unique().tolist()
        for universite_adi in universite_adi_list:
            if pd.isna(universite_adi):
                continue
            # degerlendirme_puani = universite_score.loc[universite_score["Universite"] == universite_adi,"Degerlendirme Puani"].values[0]
            degerlendirme_kategorisi = universite_score.loc[universite_score["Universite"] == universite_adi,"Degerlendirme Kategorisi"].values[0]
            degerlendirme_puani = universite_score.loc[universite_score["Universite"] == universite_adi,"Degerlendirme Puani"].values[0]

            df.loc[df["Universite Adi"] == universite_adi, "Universite Kategori"] = degerlendirme_kategorisi
            df.loc[df["Universite Adi"] == universite_adi, "Universite Degerlendirme Puani"] = degerlendirme_puani
        df["Universite Adi"] = df["Universite Adi"].replace("Kötü",0)
        df["Universite Adi"] = df["Universite Adi"].replace("Orta",1)
        df["Universite Adi"] = df["Universite Adi"].replace("İyi",2)
        return df
    
    def calculate_age(self,df):
        df["Basvuru Yili"] = df["Basvuru Yili"].astype(int)
        df["Dogum Tarihi"] = df["Dogum Tarihi"].astype(int)
        df["Age"] = df["Basvuru Yili"] - df["Dogum Tarihi"]
        df.drop(["Dogum Tarihi"],axis = 1, inplace = True)
        df = df[(df['Age'] >= 18) & (df['Age'] <= 25)] # 18-25
        return df

    def fix_bolum_column(self,df):
        df['Bölüm'] = df['Bölüm'].str.replace(r'\(.*?\)', '', regex=True)
        df['Bölüm'] = df['Bölüm'].str.strip()
        # "Fakülte" veya "fakülte" kelimelerini silmek için regex ile işlem yap
        df['Bölüm'] = df['Bölüm'].str.replace(r'\b[Ff]akülte(si)?\b', '', regex=True).str.strip()
        df['Bölüm'] = df['Bölüm'].str.replace(r'\b[Ff]akulte(si)?\b', '', regex=True).str.strip()

        df["Bölüm"] = df["Bölüm"].replace("İşletme - Ekonomi","İşletme")
        # df["Bölüm"] = df["Bölüm"].replace("İŞLETME","İşletme")
        # df["Bölüm"] = df["Bölüm"].replace("işletme","İşletme")

        df["Bölüm"] = df["Bölüm"].replace("Elektronik Mühendisliği","Elektrik-Elektronik Mühendisliği")
        df["Bölüm"] = df["Bölüm"].replace("Elektrik Elektronik Mühendisliği","Elektrik-Elektronik Mühendisliği")
        df["Bölüm"] = df["Bölüm"].replace("Elektrik Mühendisliği","Elektrik-Elektronik Mühendisliği")
        df["Bölüm"] = df["Bölüm"].replace("Elektrik Ve Elektronik Mühendisliği","Elektrik-Elektronik Mühendisliği")
        df["Bölüm"] = df["Bölüm"].replace("Elektrik ve Elektronik Mühendisliği","Elektrik-Elektronik Mühendisliği")
        df["Bölüm"] = df["Bölüm"].replace("ELEKTRİK ELEKTRONİK MÜHENDİSLİĞİ","Elektrik-Elektronik Mühendisliği")
        # df["Bölüm"] = df["Bölüm"].replace("Elektronik ve Haberleşme Mühendisliği","Elektrik-Elektronik Mühendisliği")

        df["Bölüm"] = df["Bölüm"].replace("Makina Mühendisliği","Makine Mühendisliği")
        df["Bölüm"] = df["Bölüm"].replace("Makine Mühendisliği(i̇ngilizce)","Makine Mühendisliği")

        df['Bölüm'] = df['Bölüm'].str.replace(r'(?i)bilgisayar mühendisliği', 'Bilgisayar Mühendisliği', regex=True)
        df['Bölüm'] = df['Bölüm'].str.replace(r'(?i)yazılım mühendisliği', 'Yazılım Mühendisliği', regex=True)
        df['Bölüm'] = df['Bölüm'].str.replace(r'(?i)yönetim bilişim sistemleri', 'Yönetim Bilişim Sistemleri', regex=True)
        df['Bölüm'] = df['Bölüm'].str.replace(r'(?i)makine mühendisliği', 'Makine Mühendisliği', regex=True)
        df['Bölüm'] = df['Bölüm'].str.replace(r'(?i)mekatronik mühendisliği', 'Mekatronik Mühendisliği', regex=True)

        df['Bölüm'] = df['Bölüm'].str.replace(r'(?i)işletme', 'İşletme', regex=True)
        df['Bölüm'] = df['Bölüm'].str.replace(r'(?i)maliye', 'Maliye', regex=True)
        df['Bölüm'] = df['Bölüm'].str.replace(r'(?i)kamu yönetimi', 'Kamu Yönetimi', regex=True)
        df['Bölüm'] = df['Bölüm'].str.replace(r'(?i)ekonometri', 'Ekonometri', regex=True)
        df['Bölüm'] = df['Bölüm'].str.replace(r'(?i)hukuk', 'Hukuk', regex=True)
        df['Bölüm'] = df['Bölüm'].str.replace(r'(?i)endüstri mühendisliği', 'Endüstri Mühendisliği', regex=True)
        df['Bölüm'] = df['Bölüm'].str.replace(r'(?i)inşaat mühendisliği', 'İnşaat Mühendisliği', regex=True)
        df['Bölüm'] = df['Bölüm'].str.replace(r'(?i)iktisat', 'İktisat', regex=True)
        df['Bölüm'] = df['Bölüm'].str.replace(r'(?i)tıp', 'Tıp', regex=True)
        df['Bölüm'] = df['Bölüm'].str.replace(r'(?i)tarih', 'Tarih', regex=True)
        df['Bölüm'] = df['Bölüm'].str.replace(r'(?i)diş hekimliği', 'Diş Hekimliği', regex=True)
        df['Bölüm'] = df['Bölüm'].str.replace(r'(?i)mimarlık', 'Mimarlık', regex=True)
        df['Bölüm'] = df['Bölüm'].str.replace(r'(?i)hemşirelik', 'Hemşirelik', regex=True)
        df['Bölüm'] = df['Bölüm'].str.replace(r'(?i)uluslararası ilişkiler', 'Uluslararası İlişkiler', regex=True)
        df['Bölüm'] = df['Bölüm'].str.replace(r'(?i)siyaset bilimi ve kamu yönetimi', 'Siyaset Bilimi ve Kamu Yönetimi', regex=True)


        df['Bölüm'] = df['Bölüm'].str.replace(r'(?i)fizyoterapi ve rehabilitasyon', 'Fizyoterapi ve Rehabilitasyon', regex=True)

        df['Bölüm'] = df['Bölüm'].str.replace(r'(?i).*mimarlık.*', 'Mimarlık', regex=True)
        df['Bölüm'] = df['Bölüm'].str.replace(r'(?i).*psikoloji.*', 'Psikoloji', regex=True)
        df['Bölüm'] = df['Bölüm'].str.replace(r'(?i).*matematik.*', 'Matematik', regex=True)
        df['Bölüm'] = df['Bölüm'].str.replace(r'(?i).*kimya.*', 'Kimya', regex=True)
        df['Bölüm'] = df['Bölüm'].str.replace(r'(?i).*biyoloji.*', 'Biyoloji', regex=True)
        df['Bölüm'] = df['Bölüm'].str.replace(r'(?i).*ticaret.*', 'Ticaret', regex=True)
        df['Bölüm'] = df['Bölüm'].str.replace(r'(?i).*elektronik.*', 'Elektronik', regex=True)
        df['Bölüm'] = df['Bölüm'].str.replace(r'(?i).*deniz.*', 'Deniz', regex=True)
        df['Bölüm'] = df['Bölüm'].str.replace(r'(?i).*tarım.*', 'Tarım', regex=True)
        df['Bölüm'] = df['Bölüm'].str.replace(r'(?i).*gıda.*', 'Tarım', regex=True)
        # df['Bölüm'] = df['Bölüm'].str.replace(r'(?i).*bilgisayar.*', 'Yazılım', regex=True)
        df['Bölüm'] = df['Bölüm'].str.replace(r'(?i).*yazılım.*', 'Yazılım', regex=True)
        # df["Bölüm"] = df["Bölüm"].replace("Tıp(Cerrahpaşa)","Tıp")

        # Bilişim Sistemleri Mühendisliği
        df["Bölüm"] = df["Bölüm"].replace("Bilişim Sistemleri Mühendisliği","Yazılım")
        df["Bölüm"] = df["Bölüm"].replace("Yönetim Bilişim Sistemleri","Yazılım")
        df["Bölüm"] = df["Bölüm"].replace("Siyaset Bilimi ve Uluslararası İlişkiler","Uluslararası İlişkiler")
        df["Bölüm"] = df["Bölüm"].replace("Siyaset Bilimi ve Kamu Yönetimi","Uluslararası İlişkiler")
        # "Öğretmen" kelimesini içeren tüm değerleri "Öğretmen" olarak değiştir
        df.loc[df['Bölüm'].str.contains('Öğretmen', case=False, na=False), 'Bölüm'] = 'Öğretmen'
        df.loc[df['Bölüm'].str.contains('Edebiyat', case=False, na=False), 'Bölüm'] = 'Edebiyat'


        # df["Bölüm"] = df["Bölüm"].replace("Yazılım","Mühendislik ve Teknoloji")
        # df["Bölüm"] = df["Bölüm"].replace("Elektronik","Mühendislik ve Teknoloji")

        # Mühendislik ve Teknoloji

        df["Bölüm"] = df["Bölüm"].replace("Makine Mühendisliği","Mühendislik ve Teknoloji")
        df["Bölüm"] = df["Bölüm"].replace("Mekatronik Mühendisliği","Mühendislik ve Teknoloji")
        df["Bölüm"] = df["Bölüm"].replace("İnşaat Mühendisliği","Mühendislik ve Teknoloji")
        df["Bölüm"] = df["Bölüm"].replace("Metalurji ve Malzeme Mühendisliği","Mühendislik ve Teknoloji")
        df["Bölüm"] = df["Bölüm"].replace("Biyomedikal Mühendisliği","Mühendislik ve Teknoloji")
        df["Bölüm"] = df["Bölüm"].replace("Tekstil Mühendisliği","Mühendislik ve Teknoloji")
        df["Bölüm"] = df["Bölüm"].replace("Maden Mühendisliği","Mühendislik ve Teknoloji")
        df["Bölüm"] = df["Bölüm"].replace("Havacılık ve Uzay Mühendisliği","Mühendislik ve Teknoloji")
        df["Bölüm"] = df["Bölüm"].replace("Enerji Sistemleri Mühendisliği","Mühendislik ve Teknoloji")
        df["Bölüm"] = df["Bölüm"].replace("Biyomühendislik","Mühendislik ve Teknoloji")
        df["Bölüm"] = df["Bölüm"].replace("Çevre Mühendisliği","Mühendislik ve Teknoloji")
        df["Bölüm"] = df["Bölüm"].replace("Jeoloji Mühendisliği","Mühendislik ve Teknoloji")
        df["Bölüm"] = df["Bölüm"].replace("Gemi İnşaatı ve Gemi Makineleri Mühendisliği","Mühendislik ve Teknoloji")
        df["Bölüm"] = df["Bölüm"].replace("Uçak Mühendisliği","Mühendislik ve Teknoloji")
        df["Bölüm"] = df["Bölüm"].replace("Otomotiv Mühendisliği","Mühendislik ve Teknoloji")
        df["Bölüm"] = df["Bölüm"].replace("Kontrol ve Otomasyon Mühendisliği","Mühendislik ve Teknoloji")
        df["Bölüm"] = df["Bölüm"].replace("Harita Mühendisliği","Mühendislik ve Teknoloji")
        df["Bölüm"] = df["Bölüm"].replace("Malzeme Bilimi ve Mühendisliği","Mühendislik ve Teknoloji")
        # Bilim ve Matematik
        df["Bölüm"] = df["Bölüm"].replace("Matematik","Bilim ve Matematik")
        df["Bölüm"] = df["Bölüm"].replace("Kimya","Bilim ve Matematik")
        df["Bölüm"] = df["Bölüm"].replace("Biyoloji","Bilim ve Matematik")
        df["Bölüm"] = df["Bölüm"].replace("Fizik","Bilim ve Matematik")
        df["Bölüm"] = df["Bölüm"].replace("İstatistik","Bilim ve Matematik")
        # Sağlık
        # df["Bölüm"] = df["Bölüm"].replace("Tıp","Sağlık ve Tıp")
        df["Bölüm"] = df["Bölüm"].replace("Hemşirelik","Sağlık")
        df["Bölüm"] = df["Bölüm"].replace("Diş Hekimliği","Sağlık")
        df["Bölüm"] = df["Bölüm"].replace("Eczacılık","Sağlık")
        df["Bölüm"] = df["Bölüm"].replace("İlk ve Acil Yardım","Sağlık")
        df["Bölüm"] = df["Bölüm"].replace("Sağlık Yönetimi","Sağlık")
        df["Bölüm"] = df["Bölüm"].replace("Fizyoterapi ve Rehabilitasyon","Sağlık")
        df["Bölüm"] = df["Bölüm"].replace("Beslenme ve Diyetetik","Sağlık")
        df["Bölüm"] = df["Bölüm"].replace("Acil Yardım ve Afet Yönetimi","Sağlık")
        # İktisadi Bilimler
        df["Bölüm"] = df["Bölüm"].replace("Ekonomi ve Finans","İktisadi Bilimler")
        df["Bölüm"] = df["Bölüm"].replace("Ticaret","İktisadi Bilimler")
        df["Bölüm"] = df["Bölüm"].replace("İngilizce İşletme","İktisadi Bilimler")
        df["Bölüm"] = df["Bölüm"].replace("İngilizce İktisat","İktisadi Bilimler")
        df["Bölüm"] = df["Bölüm"].replace("İşletme","İktisadi Bilimler")
        df["Bölüm"] = df["Bölüm"].replace("İşletme Mühendisliği" ,"İktisadi Bilimler")
        df["Bölüm"] = df["Bölüm"].replace("Maliye","İktisadi Bilimler")
        df["Bölüm"] = df["Bölüm"].replace("İktisat","İktisadi Bilimler")
        df["Bölüm"] = df["Bölüm"].replace("Ekonomi","İktisadi Bilimler")
        df["Bölüm"] = df["Bölüm"].replace("Bankacılık ve Finans","İktisadi Bilimler")
        df["Bölüm"] = df["Bölüm"].replace("Bankacılık ve Sigortacılık","İktisadi Bilimler")
        df["Bölüm"] = df["Bölüm"].replace("Girişimcilik","İktisadi Bilimler")
        df["Bölüm"] = df["Bölüm"].replace("Turizm İşletmeciliği","İktisadi Bilimler")
        df["Bölüm"] = df["Bölüm"].replace("Ekonometri","İktisadi Bilimler")
        df["Bölüm"] = df["Bölüm"].replace("Kamu Yönetimi","İktisadi Bilimler")
        df["Bölüm"] = df["Bölüm"].replace("Çalışma Ekonomisi ve Endüstri İlişkileri","İktisadi Bilimler")
        # Sosyal Bilimler
        df["Bölüm"] = df["Bölüm"].replace("Psikoloji","Sosyal Bilimler")
        df["Bölüm"] = df["Bölüm"].replace("Sosyoloji","Sosyal Bilimler")
        df["Bölüm"] = df["Bölüm"].replace("Felsefe","Sosyal Bilimler")
        df["Bölüm"] = df["Bölüm"].replace("Sosyal Hizmet","Sosyal Bilimler")
        df["Bölüm"] = df["Bölüm"].replace("Tarih","Sosyal Bilimler")
        df["Bölüm"] = df["Bölüm"].replace("Coğrafya","Sosyal Bilimler")
        df["Bölüm"] = df["Bölüm"].replace("Edebiyat","Sosyal Bilimler")
        df["Bölüm"] = df["Bölüm"].replace("Arkeoloji","Sosyal Bilimler")
        df["Bölüm"] = df["Bölüm"].replace("Çeviribilim","Sosyal Bilimler")
        df["Bölüm"] = df["Bölüm"].replace("Halkla İlişkiler ve Tanıtım","Sosyal Bilimler")
        df["Bölüm"] = df["Bölüm"].replace("Sanat Tarihi","Sosyal Bilimler")
        df["Bölüm"] = df["Bölüm"].replace("Mütercim-Tercümanlık","Sosyal Bilimler")
        df["Bölüm"] = df["Bölüm"].replace("Halkla İlişkiler ve Reklamcılık","Sosyal Bilimler")
        df["Bölüm"] = df["Bölüm"].replace("Gazetecilik","Sosyal Bilimler")
        # Eğitim ve Öğretim
        df["Bölüm"] = df["Bölüm"].replace("Öğretmen","Eğitim ve Öğretim")
        # İslami İlimler
        df["Bölüm"] = df["Bölüm"].replace("İlahiyat","İslami İlimler")
        # Diğer 
        # 
        df["Bölüm"] = df["Bölüm"].replace("Çocuk Gelişimi","Diğer")
        df["Bölüm"] = df["Bölüm"].replace("Deniz","Diğer")
        df["Bölüm"] = df["Bölüm"].replace("Ebelik","Diğer")
        df["Bölüm"] = df["Bölüm"].replace("Şehir ve Bölge Planlama","Diğer")
        df["Bölüm"] = df["Bölüm"].replace("Endüstri Ürünleri Tasarımı","Diğer")
        df["Bölüm"] = df["Bölüm"].replace("Radyo, Televizyon ve Sinema","Diğer")
        # Tarım ve İslami İlimlerin Diğer'e dönüştürülmesi
        df["Bölüm"] = df["Bölüm"].replace("Tarım","Diğer")
        df["Bölüm"] = df["Bölüm"].replace("İslami İlimler","Diğer")
        frequency = 50
        value_counts = df["Bölüm"].value_counts()
        to_drop = value_counts[value_counts < frequency].index
        df = df[~df["Bölüm"].isin(to_drop)]

        # Her bir bölümün frekansını hesaplama
        # Frekansı 100'den düşük olan değerleri "Diğer" olarak değiştirme
        frequencies = df['Bölüm'].value_counts()
        df['Bölüm'] = df['Bölüm'].apply(lambda x: 'Diğer' if pd.notna(x) and frequencies.get(x, 0) < 100 else x)

        columns_to_convert = []
        unique_elements = df["Bölüm"].unique()
        for unique in unique_elements:
            if pd.isna(unique):
                continue
            columns_to_convert.append(f"sektor_{unique}")
        df = pd.get_dummies(df, columns=["Bölüm"], prefix="sektor")
        df[columns_to_convert] = df[columns_to_convert].astype(int)

        return df
    def run_process(self):
        df = self.raw_data
        df = self.fix_cinsiyet_column(df)
        df = self.fix_dogum_tarihi(df)
        df = self.fix_dogum_yeri_column(df)
        df = self.fix_bolum_column(df)
        # df = self.fix_sehir_sutuns(df,"Ikametgah Sehri")
        df = self.fix_universite_turu_column(df)
        df = self.fix_burs_aliyor_mu_column(df)
        df = self.fix_universite_kacinci_sinif_column(df)

        df = self.fix_lise_turu_column(df)
    
        df = self.fix_lise_bolumu_column(df)
        df = self.fix_lise_mezuniyet_notu_column(df)
        df = self.fix_baska_bir_kurumdan_burs_aliyor_mu_column(df)

        df = self.fix_egitim_durumu(df,"Anne Egitim Durumu")
        df = self.fix_egitim_durumu(df,"Baba Egitim Durumu")

        df = self.fix_calisma_durumu_column(df,'Anne Calisma Durumu')
        df = self.fix_calisma_durumu_column(df,"Baba Calisma Durumu")

        df = self.fix_sektor_column(df,"Anne Sektor")
        df = self.fix_sektor_column(df,"Baba Sektor")
        df = self.fix_kardes_sayisi_column(df)
        df = self.fix_girisimcilik_kulupleri_column(df)
        df = self.fix_profesyonel_spor_dali_column(df)

        df = self.fix_stk_projesi_column(df)
        df = self.fix_stk_uyesi_column(df)
        
        df = self.fix_girisimcilik_deneyim_column(df)
        df = self.fix_ingilizce_biliyor_musunuz_column(df)
        df = self.fix_universite_not_ortalamasi_column(df)
        df = self.fix_universite_adi_column(df)
        df = self.calculate_age(df)
        columns_to_dropped = self.drop_high_nan_columns(df,30) # 30
        return df,columns_to_dropped