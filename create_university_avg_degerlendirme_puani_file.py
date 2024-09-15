from train_data_processor import TrainDataProcessor
import pandas as pd

def format_university_name(university_name):
    if pd.isna(university_name):
        return university_name
    # Üniversite ismini title case yap
    formatted_name = university_name.title()
    # Türkçe karakterlerde hatalı dönüşüm olmaması için düzeltme
    formatted_name = formatted_name.replace("İ", "İ").replace("Ü", "Ü").replace("Ö", "Ö").replace("Ğ", "Ğ").replace("Ç", "Ç").replace("Ş", "Ş")
    return formatted_name

def create_university_avg_degerlendirme_puani_file(df):
    universite_adi_list = df["Universite Adi"].unique().tolist()
    uni_df = pd.DataFrame(columns = ["Universite","Degerlendirme Puani"])

    for universite in universite_adi_list:
        if pd.isna(universite):
            continue
        temp = df.loc[df["Universite Adi"] == universite]
        degerlendirme_puani_avg = temp["Degerlendirme Puani"].sum() / temp.shape[0]
        # uni_df.loc[len(uni_df)] = [format_university_name(universite),degerlendirme_puani_avg]
        uni_df.loc[len(uni_df)] = [universite,degerlendirme_puani_avg]
        # universite_average_degerlendirme_puani_dict.update({format_university_name(universite):degerlendirme_puani_avg})
    # Aralıkları ve etiketleri belirleme
    bins = [0, 30, 60, float('inf')]  # 30'dan büyük olanlar için inf kullanılır
    labels = ['Kötü', 'Orta', 'İyi']

    # Kategorilere göre güncelleme
    uni_df["Degerlendirme Kategorisi"] = pd.cut(uni_df["Degerlendirme Puani"], bins=bins, labels=labels, right=False)
    uni_df.to_csv("data/university_avg_degerlendirme_puani.csv",index = False)
    return uni_df

file_path = "data/train.csv"
data_processor = TrainDataProcessor(file_path)

df,columns_to_dropped = data_processor.run_process()

df["Universite Adi"] = df["Universite Adi"].apply(format_university_name)
uni_df = create_university_avg_degerlendirme_puani_file(df)