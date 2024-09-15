import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import re
import numpy as np

class TestDataProcessor:
    def __init__(self,test_file_path):
        self.test_data = pd.read_csv(test_file_path)

    def drop_high_nan_columns(self,df,threshold_nan_ratio=30):
        nan_ratios = df.isna().mean() * 100
        # NaN oranı %50'den fazla olan sütunları filtrele
        columns_with_high_nan = nan_ratios[nan_ratios > threshold_nan_ratio].index
        df = df.drop(columns=columns_with_high_nan)
        return df

    def fix_cinsiyet_column(self,df):
        df['Cinsiyet'] = df['Cinsiyet'].replace('Belirtmek istemiyorum', 'Erkek')
        df['Cinsiyet'] = df['Cinsiyet'].replace('Erkek', 1)
        df['Cinsiyet'] = df['Cinsiyet'].replace('Kadın', 0)
        return df

    def fix_dogum_tarihi(self,df):
        print("before",df.shape[0])
        df['Dogum Tarihi'] = pd.to_datetime(df['Dogum Tarihi'], format='%d.%m.%Y %H:%M')
        df['Dogum Tarihi'] = df['Dogum Tarihi'].dt.year
        df['Dogum Tarihi'] = df['Dogum Tarihi'].astype(int)
        print("after",df.shape[0])
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
        columns_to_convert = []
        unique_elements = df["Universite Kacinci Sinif"].unique()
        for unique in unique_elements:
            if pd.isna(unique):
                continue
            columns_to_convert.append(f"is_{unique}")

        df = pd.get_dummies(df, columns=["Universite Kacinci Sinif"], prefix="is")
        df[columns_to_convert] = df[columns_to_convert].astype(int)
        return df

    def fix_lise_turu_column(self,df):
        df["Lise Turu"] = df['Lise Turu'].replace('Özel', 1)
        df["Lise Turu"] = df['Lise Turu'].replace('Devlet', 0)
        return df

    def fix_lise_bolumu(self,df):
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
        df["Lise Mezuniyet Notu"] = df["Lise Mezuniyet Notu"].replace("25 - 49",0)
        df["Lise Mezuniyet Notu"] = df["Lise Mezuniyet Notu"].replace("50 - 74",1)
        df["Lise Mezuniyet Notu"] = df["Lise Mezuniyet Notu"].replace("75 - 100",2)
        return df

    def fix_baska_bir_kurumdan_burs_aliyor_mu_column(self,df):
        df["Baska Bir Kurumdan Burs Aliyor mu?"] = df["Baska Bir Kurumdan Burs Aliyor mu?"].replace("Hayır",0)
        df["Baska Bir Kurumdan Burs Aliyor mu?"] = df["Baska Bir Kurumdan Burs Aliyor mu?"].replace("Evet",1)
        return df

    def fix_egitim_durumu(self,df,column_name):
        if "Anne" in column_name:
            column_type = "anne"
        if "Baba" in column_name:
            column_type = "baba"

        df[column_name] = df[column_name].replace('Doktora', 'Üniversite')
        df[column_name] = df[column_name].replace('Yüksek Lisans', 'Üniversite')
        df[column_name] = df[column_name].replace('İlkokul', 'Eğitimi yok')
        df[column_name] = df[column_name].replace('Ortaokul', 'Eğitimi yok')

        columns_to_convert = []
        unique_elements = df[column_name].unique()
        for unique in unique_elements:
            if pd.isna(unique):
                continue
            columns_to_convert.append(f"{column_type}_{unique}")

        df = pd.get_dummies(df, columns=[column_name], prefix=column_type)
        df[columns_to_convert] = df[columns_to_convert].astype(int)
        return df

    def fix_calisma_durumu(self,df,column_name):
        df[column_name] = df[column_name].replace("Evet",1)
        df[column_name] = df[column_name].replace("Hayır",0)
        return df

    def fix_kardes_sayisi_column(self,df):
        df["Kardes Sayisi"] = df["Kardes Sayisi"].astype(float)
        return df

    def fix_girisimcilik_kulupleri_column(self,df):
        df["Girisimcilik Kulupleri Tarzi Bir Kulube Uye misiniz?"] = df["Girisimcilik Kulupleri Tarzi Bir Kulube Uye misiniz?"].replace("Hayır",0)
        df["Girisimcilik Kulupleri Tarzi Bir Kulube Uye misiniz?"] = df["Girisimcilik Kulupleri Tarzi Bir Kulube Uye misiniz?"].replace("Evet",1)
        return df

    def fix_profesyonel_spor_dali_column(self,df):
        df["Profesyonel Bir Spor Daliyla Mesgul musunuz?"] = df["Profesyonel Bir Spor Daliyla Mesgul musunuz?"].replace("Hayır",0)
        df["Profesyonel Bir Spor Daliyla Mesgul musunuz?"] = df["Profesyonel Bir Spor Daliyla Mesgul musunuz?"].replace("Evet",1)
        return df

    def fix_stk_uyesi_column(self,df):
        df["Aktif olarak bir STK üyesi misiniz?"] = df["Aktif olarak bir STK üyesi misiniz?"].replace("Hayır",0)
        df["Aktif olarak bir STK üyesi misiniz?"] = df["Aktif olarak bir STK üyesi misiniz?"].replace("Evet",1)
        return df

    def fix_girisimcilik_deneyim_column(self,df):
        df["Girisimcilikle Ilgili Deneyiminiz Var Mi?"] = df["Girisimcilikle Ilgili Deneyiminiz Var Mi?"].replace("Evet",1)
        df["Girisimcilikle Ilgili Deneyiminiz Var Mi?"] = df["Girisimcilikle Ilgili Deneyiminiz Var Mi?"].replace("Hayır",0)
        return df

    def fix_ingilizce_biliyor_musunuz_column(self,df):
        df["Ingilizce Biliyor musunuz?"] = df["Ingilizce Biliyor musunuz?"].replace("Evet",1)
        df["Ingilizce Biliyor musunuz?"] = df["Ingilizce Biliyor musunuz?"].replace("Hayır",0)
        return df
    
    def fix_universite_not_ortalamasi_column(self,df):
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
        count_new_250_299 = int(count_hazirligim * ratio_250_299) + 2
        count_new_300_349 = int(count_hazirligim * ratio_300_349)
        count_new_350_400 = int(count_hazirligim * ratio_350_400)
        new_values = (['0 - 1.79'] * count_new_0_179 + ['1.80 - 2.49'] * count_new_180_249 + ["2.50 - 2.99"] * count_new_250_299 + ["3.00 - 3.49"] * count_new_300_349 + ["3.50 - 4.00"] * count_new_350_400)
        np.random.shuffle(new_values)
        indices_hazirliğim = df.loc[df['Universite Not Ortalamasi'] == 'Hazırlığım'].index.tolist()
        df.loc[indices_hazirliğim, 'Universite Not Ortalamasi'] = new_values

        df["Universite Not Ortalamasi"] = df["Universite Not Ortalamasi"].replace("0 - 1.79",0)
        df["Universite Not Ortalamasi"] = df["Universite Not Ortalamasi"].replace("1.80 - 2.49",1)
        df["Universite Not Ortalamasi"] = df["Universite Not Ortalamasi"].replace("2.50 - 2.99",2)
        df["Universite Not Ortalamasi"] = df["Universite Not Ortalamasi"].replace("3.00 - 3.49",3)
        df["Universite Not Ortalamasi"] = df["Universite Not Ortalamasi"].replace("3.50 - 4.00",4)
        return df

    def fix_sehir_column(self,df):
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
        df['Dogum Yeri'] = df['Dogum Yeri'].apply(custom_lowercase)
        df['Dogum Yeri'] = df['Dogum Yeri'].replace("iğdır","ığdır")

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

    def fix_universite_adi_column(self,df):
        universite_score_file = "data/university_avg_degerlendirme_puani.csv"
        universite_score = pd.read_csv(universite_score_file)
        universite_adi_list = df["Universite Adi"].unique().tolist()
        for universite_adi in universite_adi_list:
            try:
                if pd.isna(universite_adi):
                    continue
                # degerlendirme_puani = universite_score.loc[universite_score["Universite"] == universite_adi,"Degerlendirme Puani"].values[0]
                degerlendirme_kategorisi = universite_score.loc[universite_score["Universite"] == universite_adi,"Degerlendirme Kategorisi"].values[0]
                degerlendirme_puani = universite_score.loc[universite_score["Universite"] == universite_adi,"Degerlendirme Puani"].values[0]

                df.loc[df["Universite Adi"] == universite_adi, "Universite Kategori"] = degerlendirme_kategorisi
                df.loc[df["Universite Adi"] == universite_adi, "Universite Degerlendirme Puani"] = degerlendirme_puani
            except:
                pass
        ## BURASI DÜZELECEK.
        df.loc[df["Universite Adi"] == "TED Üniversitesi","Universite Kategori"] = "İyi"
        df.loc[df["Universite Adi"] == "TOBB Ekonomi ve Teknoloji Üniversitesi","Universite Kategori"] = "İyi"
        df.loc[df["Universite Adi"] == "MEF Üniversitesi","Universite Kategori"] = "Orta"
        df.loc[df["Universite Adi"] == "KKTC'de üniversite okuyorum.","Universite Kategori"] = "Kötü"
        df.loc[df["Universite Adi"] == "Konya Gıda ve Tarım Üniversitesi","Universite Kategori"] = "Kötü"
        df.loc[df["Universite Adi"] == "Adana Alparslan Türkeş Bilim ve Teknoloji Üniversitesi","Universite Kategori"] = "Kötü"
        df.loc[df["Universite Adi"] == "Gaziantep İslam Bilim ve Teknoloji Üniversitesi","Universite Kategori"] = "Kötü"
        df.loc[df["Universite Adi"] == "Yurtdışında üniversite okuyorum.","Universite Kategori"] = "Orta"
        df.loc[df["Universite Adi"] == "KTO Karatay Üniversitesi","Universite Kategori"] = "Kötü"
        df.loc[df["Universite Adi"] == "İstanbul Sağlık ve Teknoloji Üniversitesi","Universite Kategori"] = "Kötü"
        df.loc[df["Universite Adi"] == "Konya Gıda ve Tarım Üniversites","Universite Kategori"] = "Kötü"
        df.loc[df["Universite Adi"] == "Kocaeli Sağlık ve Teknoloji Üniversitesi","Universite Kategori"] = "Kötü"
        df.loc[df["Universite Adi"] == "Sivas Bilim ve Teknoloji Üniversitesi","Universite Kategori"] = "Kötü"
        df.loc[df["Universite Adi"] == "Ankara Müzik ve Güzel Sanatlar Üniversitesi","Universite Kategori"] = "Kötü"
        # df["Universite Adi"] = df["Universite Adi"].replace("TED Üniversitesi","İyi")
        # df["Universite Adi"] = df["Universite Adi"].replace("TOBB Ekonomi ve Teknoloji Üniversitesi","İyi")
        # df["Universite Adi"] = df["Universite Adi"].replace("MEF Üniversitesi","Orta")
        # df["Universite Adi"] = df["Universite Adi"].replace("KKTC'de üniversite okuyorum.","Kötü")
        # df["Universite Adi"] = df["Universite Adi"].replace("Adana Alparslan Türkeş Bilim ve Teknoloji Üniversitesi","Kötü")
        # df["Universite Adi"] = df["Universite Adi"].replace("Gaziantep İslam Bilim ve Teknoloji Üniversitesi","Kötü")
        # df["Universite Adi"] = df["Universite Adi"].replace("Yurtdışında üniversite okuyorum.","Orta")
        # df["Universite Adi"] = df["Universite Adi"].replace("KTO Karatay Üniversitesi","Kötü")
        # df["Universite Adi"] = df["Universite Adi"].replace("İstanbul Sağlık ve Teknoloji Üniversitesi","Kötü")
        # df["Universite Adi"] = df["Universite Adi"].replace("Konya Gıda ve Tarım Üniversitesi","Kötü")
        # df["Universite Adi"] = df["Universite Adi"].replace("Kocaeli Sağlık ve Teknoloji Üniversitesi","Kötü")
        # df["Universite Adi"] = df["Universite Adi"].replace("Sivas Bilim ve Teknoloji Üniversitesi","Kötü")
        # df["Universite Adi"] = df["Universite Adi"].replace("Ankara Müzik ve Güzel Sanatlar Üniversitesi","Kötü")
        # df["Universite Adi"] = df["Universite Adi"].replace("Kötü",0)
        # df["Universite Adi"] = df["Universite Adi"].replace("Orta",1)
        # df["Universite Adi"] = df["Universite Adi"].replace("İyi",2)
        return df
    
    def calculate_age(self,df):
        df["Basvuru Yili"] = df["Basvuru Yili"].astype(int)
        df["Dogum Tarihi"] = df["Dogum Tarihi"].astype(int)
        df["Age"] = df["Basvuru Yili"] - df["Dogum Tarihi"]
        df.drop(["Dogum Tarihi"],axis = 1, inplace = True)
        return df
    
    def fix_bolum_column(self,df):
        df.loc[df['Bölüm'].str.contains('Öğretmen', case=False, na=False), 'Bölüm'] = 'Öğretmen'
        df.loc[df['Bölüm'].str.contains('İşletme', case=False, na=False), 'Bölüm'] = 'İşletme'
        df.loc[df['Bölüm'].str.contains('Finans', case=False, na=False), 'Bölüm'] = 'İktisadi Bilimler'
        df.loc[df['Bölüm'].str.contains('Banka', case=False, na=False), 'Bölüm'] = 'İktisadi Bilimler'
        df.loc[df['Bölüm'].str.contains('Muhasebe', case=False, na=False), 'Bölüm'] = 'İktisadi Bilimler'
        df.loc[df['Bölüm'].str.contains('Edebiyat', case=False, na=False), 'Bölüm'] = 'Edebiyat'
        df.loc[df['Bölüm'].str.contains('Bitki', case=False, na=False), 'Bölüm'] = 'Tarım'
        df.loc[df['Bölüm'].str.contains('Toprak', case=False, na=False), 'Bölüm'] = 'Tarım'
        df.loc[df['Bölüm'].str.contains('Bilişim', case=False, na=False), 'Bölüm'] = 'Yazılım'
        df.loc[df['Bölüm'].str.contains('Mütercim', case=False, na=False), 'Bölüm'] = 'Diğer'
        df.loc[df['Bölüm'].str.contains('Radyo', case=False, na=False), 'Bölüm'] = 'Diğer'
        df.loc[df['Bölüm'].str.contains('Televizyon', case=False, na=False), 'Bölüm'] = 'Diğer'
        df.loc[df['Bölüm'].str.contains('Lojistik', case=False, na=False), 'Bölüm'] = 'Diğer'
        df.loc[df['Bölüm'].str.contains('Eğitim', case=False, na=False), 'Bölüm'] = 'Eğitim ve Öğretim'
        df.loc[df['Bölüm'].str.contains('Tarih', case=False, na=False), 'Bölüm'] = 'Tarih'

        df['Bölüm'] = df['Bölüm'].str.replace(r'(?i).*mimarlık.*', 'Mimarlık', regex=True)
        df['Bölüm'] = df['Bölüm'].str.replace(r'(?i).*psikoloji.*', 'Psikoloji', regex=True)
        df['Bölüm'] = df['Bölüm'].str.replace(r'(?i).*matematik.*', 'Matematik', regex=True)
        df['Bölüm'] = df['Bölüm'].str.replace(r'(?i).*kimya.*', 'Kimya', regex=True)
        df['Bölüm'] = df['Bölüm'].str.replace(r'(?i).*fizik.*', 'Fizik', regex=True)
        df['Bölüm'] = df['Bölüm'].str.replace(r'(?i).*biyoloji.*', 'Biyoloji', regex=True)
        df['Bölüm'] = df['Bölüm'].str.replace(r'(?i).*tarım.*', 'Tarım', regex=True)
        df['Bölüm'] = df['Bölüm'].str.replace(r'(?i).*elektronik.*', 'Elektronik', regex=True)
        df['Bölüm'] = df['Bölüm'].str.replace(r'(?i).*spor.*', 'Spor', regex=True)

        #'İktisadi Bilimler'
        df["Bölüm"] = df["Bölüm"].replace("İşletme","İktisadi Bilimler")
        df["Bölüm"] = df["Bölüm"].replace("İktisat","İktisadi Bilimler")
        df["Bölüm"] = df["Bölüm"].replace("Maliye","İktisadi Bilimler")
        df["Bölüm"] = df["Bölüm"].replace("Ekonomi","İktisadi Bilimler")
        df["Bölüm"] = df["Bölüm"].replace("İşletme Mühendisliği" ,"İktisadi Bilimler")
        df["Bölüm"] = df["Bölüm"].replace("Çalışma Ekonomisi ve Endüstri İlişkileri","İktisadi Bilimler")
        df["Bölüm"] = df["Bölüm"].replace("Uluslararası Ticaret ve İşletmecilik","İktisadi Bilimler")
        df["Bölüm"] = df["Bölüm"].replace("Ekonometri","İktisadi Bilimler")
        df["Bölüm"] = df["Bölüm"].replace("Muhasebe","İktisadi Bilimler")
        df["Bölüm"] = df["Bölüm"].replace("Kamu Yönetimi","İktisadi Bilimler")
        df["Bölüm"] = df["Bölüm"].replace("Ekonomi ve Finans","İktisadi Bilimler")
        df["Bölüm"] = df["Bölüm"].replace("Bankacılık ve Sigortacılık","İktisadi Bilimler")

        # 
        #'Mühendislik ve Teknoloji'
        df["Bölüm"] = df["Bölüm"].replace("İnşaat Mühendisliği","Mühendislik ve Teknoloji")
        df["Bölüm"] = df["Bölüm"].replace("Makine Mühendisliği","Mühendislik ve Teknoloji")
        df["Bölüm"] = df["Bölüm"].replace("Mekatronik Mühendisliği","Mühendislik ve Teknoloji")
        df["Bölüm"] = df["Bölüm"].replace("Metalurji ve Malzeme Mühendisliği","Mühendislik ve Teknoloji")
        df["Bölüm"] = df["Bölüm"].replace("Gıda Mühendisliği","Mühendislik ve Teknoloji")
        df["Bölüm"] = df["Bölüm"].replace("Biyomühendislik","Mühendislik ve Teknoloji")
        df["Bölüm"] = df["Bölüm"].replace("Çevre Mühendisliği","Mühendislik ve Teknoloji")
        df["Bölüm"] = df["Bölüm"].replace("Biyomedikal Mühendisliği","Mühendislik ve Teknoloji")
        df["Bölüm"] = df["Bölüm"].replace("Havacılık ve Uzay Mühendisliği","Mühendislik ve Teknoloji")
        df["Bölüm"] = df["Bölüm"].replace("Kontrol ve Otomasyon Mühendisliği","Mühendislik ve Teknoloji")
        df["Bölüm"] = df["Bölüm"].replace("Uçak Mühendisliği","Mühendislik ve Teknoloji")
        df["Bölüm"] = df["Bölüm"].replace("Tekstil Mühendisliği","Mühendislik ve Teknoloji")
        df["Bölüm"] = df["Bölüm"].replace("Enerji Sistemleri Mühendisliği","Mühendislik ve Teknoloji")
        df["Bölüm"] = df["Bölüm"].replace("Uzay Mühendisliği","Mühendislik ve Teknoloji")
        df["Bölüm"] = df["Bölüm"].replace("Maden Mühendisliği","Mühendislik ve Teknoloji")

        # Kontrol ve Otomasyon Mühendisliği
        #'Sağlık'
        df["Bölüm"] = df["Bölüm"].replace("Hemşirelik","Sağlık")
        df["Bölüm"] = df["Bölüm"].replace("Diş Hekimliği","Sağlık")
        df["Bölüm"] = df["Bölüm"].replace("Eczacılık","Sağlık")
        df["Bölüm"] = df["Bölüm"].replace("İlk ve Acil Yardım","Sağlık")
        df["Bölüm"] = df["Bölüm"].replace("Sağlık Yönetimi","Sağlık")
        df["Bölüm"] = df["Bölüm"].replace("Fizyoterapi ve Rehabilitasyon","Sağlık")
        df["Bölüm"] = df["Bölüm"].replace("Beslenme ve Diyetetik","Sağlık")
        df["Bölüm"] = df["Bölüm"].replace("Acil Yardım ve Afet Yönetimi","Sağlık")
        df["Bölüm"] = df["Bölüm"].replace("Dil ve Konuşma Terapisi","Sağlık")
        df["Bölüm"] = df["Bölüm"].replace("Hemşirelik ve Sağlık Hizmetleri","Sağlık")
        df["Bölüm"] = df["Bölüm"].replace("Odyoloji","Sağlık")
        #'Tıp'
        #'Sosyal Bilimler' 
        df["Bölüm"] = df["Bölüm"].replace("Psikoloji","Sosyal Bilimler")
        df["Bölüm"] = df["Bölüm"].replace("Sosyoloji","Sosyal Bilimler")
        df["Bölüm"] = df["Bölüm"].replace("Sosyal Hizmet","Sosyal Bilimler")
        df["Bölüm"] = df["Bölüm"].replace("Tarih","Sosyal Bilimler")
        df["Bölüm"] = df["Bölüm"].replace("Coğrafya","Sosyal Bilimler")
        df["Bölüm"] = df["Bölüm"].replace("Edebiyat","Sosyal Bilimler")
        df["Bölüm"] = df["Bölüm"].replace("Halkla İlişkiler ve Reklamcılık","Sosyal Bilimler")
        df["Bölüm"] = df["Bölüm"].replace("Felsefe","Sosyal Bilimler")
        df["Bölüm"] = df["Bölüm"].replace("Gazetecilik","Sosyal Bilimler")
        df["Bölüm"] = df["Bölüm"].replace("Halkla İlişkiler ve Tanıtım","Sosyal Bilimler")
        df["Bölüm"] = df["Bölüm"].replace("İş Sağlığı ve Güvenliği","Sosyal Bilimler")
        #'Yazılım' 
        # df["Bölüm"] = df["Bölüm"].replace("Bilgisayar Mühendisliği","Yazılım")
        df["Bölüm"] = df["Bölüm"].replace("Yazılım Mühendisliği","Yazılım")
        df["Bölüm"] = df["Bölüm"].replace("Yönetim Bilişim Sistemleri","Yazılım")
        df["Bölüm"] = df["Bölüm"].replace("Bilgisayar Bilimi ve Mühendisliği","Yazılım")
        df["Bölüm"] = df["Bölüm"].replace("Bilgisayar Teknolojisi ve Bilişim Sistemleri","Yazılım")
        df["Bölüm"] = df["Bölüm"].replace("Bilgisayar Bilimleri","Yazılım")
        df["Bölüm"] = df["Bölüm"].replace("Bilişim Sistemleri ve Teknolojileri","Yazılım")
        df["Bölüm"] = df["Bölüm"].replace("Bilgisayar ve Yazılım Mühendisliği","Yazılım")
        df["Bölüm"] = df["Bölüm"].replace("Dijital Oyun Tasarımı","Yazılım")
        # Dijital Oyun Tasarımı
        #Bilişim Sistemleri ve Teknolojileri
        #'Diğer' 
        df["Bölüm"] = df["Bölüm"].replace("Çocuk Gelişimi","Diğer")
        df["Bölüm"] = df["Bölüm"].replace("Ebelik","Diğer")
        df["Bölüm"] = df["Bölüm"].replace("Gastronomi ve Mutfak Sanatları","Diğer")
        df["Bölüm"] = df["Bölüm"].replace("Veteriner","Diğer")
        df["Bölüm"] = df["Bölüm"].replace("Spor","Diğer")
        df["Bölüm"] = df["Bölüm"].replace("Antrenörlük","Diğer")
        df["Bölüm"] = df["Bölüm"].replace("Antrenörlük Eğitimi","Diğer")
        df["Bölüm"] = df["Bölüm"].replace("Radyo,Televizyon ve Sinema","Diğer")

        #'Endüstri Mühendisliği'
        #'Elektronik' 
        df["Bölüm"] = df["Bölüm"].replace("Elektrik-Elektronik Mühendisliği","Elektronik")
        df["Bölüm"] = df["Bölüm"].replace("Elektronik ve Haberleşme Mühendisliği","Elektronik")
        df["Bölüm"] = df["Bölüm"].replace("Elektrik Mühendisliği","Elektronik")
        #'Bilim ve Matematik'
        df["Bölüm"] = df["Bölüm"].replace("Matematik","Bilim ve Matematik")
        df["Bölüm"] = df["Bölüm"].replace("Kimya","Bilim ve Matematik")
        df["Bölüm"] = df["Bölüm"].replace("Biyoloji","Bilim ve Matematik")
        df["Bölüm"] = df["Bölüm"].replace("Fizik","Bilim ve Matematik")
        df["Bölüm"] = df["Bölüm"].replace("İstatistik","Bilim ve Matematik")
        df["Bölüm"] = df["Bölüm"].replace("Moleküler Biyoloji ve Genetik","Bilim ve Matematik")
        #'Uluslararası İlişkiler'
        df["Bölüm"] = df["Bölüm"].replace("Siyaset Bilimi ve Uluslararası İlişkiler","Uluslararası İlişkiler")
        df["Bölüm"] = df["Bölüm"].replace("Siyaset Bilimi ve Kamu Yönetimi","Uluslararası İlişkiler")
        df["Bölüm"] = df["Bölüm"].replace("Uluslararası Ticaret ve Finansman","Uluslararası İlişkiler")
        df["Bölüm"] = df["Bölüm"].replace("Uluslararası Ticaret","Uluslararası İlişkiler")
        df["Bölüm"] = df["Bölüm"].replace("Uluslararası Ticaret ve Lojistik","Uluslararası İlişkiler")
        #'Hukuk'
        #'Eğitim ve Öğretim'
        df["Bölüm"] = df["Bölüm"].replace("Öğretmen","Eğitim ve Öğretim")
        #'Mimarlık'
        #'İslami İlimler'
        df["Bölüm"] = df["Bölüm"].replace("İlahiyat","İslami İlimler")
        # Tarım ve İslami İlimlerin Dönüştürülmesi
        df["Bölüm"] = df["Bölüm"].replace("Tarım","Diğer")
        df["Bölüm"] = df["Bölüm"].replace("İslami İlimler","Diğer")
        ## 
        # df.loc[df['Bölüm'].str.contains('Mühendis', case=False, na=False), 'Bölüm'] = 'Mühendislik ve Teknoloji'

        # Frekansı 100'den düşük olan değerleri "Diğer" olarak değiştirme
        frequencies = df['Bölüm'].value_counts()
        df['Bölüm'] = df['Bölüm'].apply(lambda x: 'Diğer' if pd.notna(x) and frequencies.get(x, 0) < 36 else x)

        columns_to_convert = []
        unique_elements = df["Bölüm"].unique()
        for unique in unique_elements:
            if pd.isna(unique):
                continue
            columns_to_convert.append(f"sektor_{unique}")
        df = pd.get_dummies(df, columns=["Bölüm"], prefix="sektor")
        df[columns_to_convert] = df[columns_to_convert].astype(int)

        return df
    
    def fix_sektor_column(self, df,column_name): # Anne Sektor ve/veya Baba Sektor droplanabilir.
        df[column_name] = df[column_name].replace("ÖZEL SEKTÖR","Özel Sektör")
        df[column_name] = df[column_name].replace("KAMU","Kamu")
        df[column_name] = df[column_name].replace("DİĞER","Diğer")
        df[column_name] = df[column_name].replace("-","Sektör Yok")
        # df[column_name] = df[column_name].replace("0","Sektör Yok")

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

    def run_process(self):
        df = self.test_data
        # df = self.drop_high_nan_columns(df)
        df = self.fix_cinsiyet_column(df)
        df = self.fix_dogum_tarihi(df)
        df = self.fix_bolum_column(df)
        df = self.fix_sehir_column(df)
        df = self.fix_universite_turu_column(df)
        df = self.fix_burs_aliyor_mu_column(df)
        df = self.fix_universite_kacinci_sinif_column(df)
        df = self.fix_lise_turu_column(df)
        df = self.fix_lise_bolumu(df)
        df = self.fix_lise_mezuniyet_notu_column(df)
        df = self.fix_baska_bir_kurumdan_burs_aliyor_mu_column(df)
        df = self.fix_egitim_durumu(df,"Anne Egitim Durumu")
        df = self.fix_egitim_durumu(df,"Baba Egitim Durumu")
        df = self.fix_calisma_durumu(df,"Anne Calisma Durumu")
        df = self.fix_calisma_durumu(df,"Baba Calisma Durumu")
        df = self.fix_sektor_column(df,"Anne Sektor")
        df = self.fix_sektor_column(df,"Baba Sektor")
        df = self.fix_kardes_sayisi_column(df)
        df = self.fix_girisimcilik_kulupleri_column(df)
        df = self.fix_profesyonel_spor_dali_column(df)
        df = self.fix_stk_uyesi_column(df)
        df = self.fix_girisimcilik_deneyim_column(df)
        df = self.fix_ingilizce_biliyor_musunuz_column(df)
        df = self.fix_universite_not_ortalamasi_column(df)
        df = self.fix_universite_adi_column(df)
        df = self.calculate_age(df)
        return df