import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import plotly.express as px

# --- SAYFA AYARLARI ---
st.set_page_config(
    page_title="Mobil Bankacılık Müşteri Deneyimi Paneli (G01)",
    page_icon="🏦",
    layout="wide"
)


# --- VERİ VE MODEL YÜKLEME (Açılışta yeniden eğitmez, kayıtlı modeli yükler) ---
@st.cache_data
def veri_yukle():
    df = pd.read_csv("banka_yorumlari_final.csv")
    konu_isimleri = {
        0: "Giriş ve Güncelleme Hataları",
        1: "Genel Memnuniyetsizlik ve Hesap Açılışı",
        2: "Yavaşlık ve Müşteri Hizmetleri",
        3: "Para Transferi (EFT) ve İşlem Ücretleri",
        4: "Uygulama Çökmesi ve Açılmama"
    }
    df["konu_adi"] = df["konu_id"].map(konu_isimleri)
    return df, konu_isimleri


@st.cache_resource
def model_yukle():
    return joblib.load("model.joblib")


df, KONU_ISIMLERI = veri_yukle()
paket = model_yukle()
duygu_modeli = paket["duygu_modeli"]
nmf_tfidf = paket["nmf_tfidf"]
nmf_model = paket["nmf_model"]

# Metin temizleme fonksiyonu (Canlı tahmin için)
turkce_stopwords = {
    "ve", "veya", "ile", "ama", "fakat", "lakin", "ancak", "çünkü", "de", "da",
    "ki", "mi", "mı", "mu", "mü", "bir", "bu", "şu", "o", "çok", "az", "daha",
    "en", "gibi", "için", "kadar", "sonra", "önce", "her", "hiç", "bile", "ben",
    "sen", "biz", "siz", "onlar", "ne", "neden", "nasıl", "var", "yok", "olarak", "ise"
}


def metin_temizle(metin):
    metin = str(metin).lower()
    metin = re.sub(r"http\S+|www\S+", "", metin)
    metin = re.sub(r"[^a-zçğıöşü\s]", " ", metin)
    kelimeler = [k for k in metin.split() if k not in turkce_stopwords and len(k) > 2]
    return " ".join(kelimeler)


# --- ÜST BAŞLIK VE KENAR ÇUBUĞU ---
st.title("🏦 Mobil Bankacılık Uygulamalarında Müşteri Deneyimi Analizi")
st.markdown(
    "**Ekip G01:** Fatih Güngör, Emircan Göçer, Arda Dadaklı | *Veriden Karara: Müşteri Sesi Karar Destek Sistemi*")

st.sidebar.header("🔍 Filtreler")
secilen_bankalar = st.sidebar.multiselect(
    "Karşılaştırılacak Bankaları Seçin:",
    options=df["banka_adi"].unique().tolist(),
    default=df["banka_adi"].unique().tolist()
)

df_filtre = df[df["banka_adi"].isin(secilen_bankalar)].copy()

# --- 4 SEKMELİ YAPI (3 Zorunlu Ekran + 1 İmza Özellik) ---
tab_panel, tab_analiz, tab_karar, tab_imza = st.tabs([
    "📊 Panel (KPI & Özet)",
    "📈 Analiz (Trend & Şikâyet Konuları)",
    "🤖 Karar Ekranı (Canlı Yorum Tahmini)",
    "🚀 İmza Özellik: Puan İyileştirme Simülatörü"
])

# =====================================================================
# 1. EKRAN: PANEL (KPI + ÖZET)
# =====================================================================
with tab_panel:
    st.subheader("Genel Performans Göstergeleri (KPI)")
    c1, c2, c3, c4 = st.columns(4)

    toplam_yorum = len(df_filtre)
    ort_puan = df_filtre["score"].mean()
    olumlu_oran = (df_filtre["duygu_etiketi"] == 1).mean() * 100
    olumsuz_oran = (df_filtre["duygu_etiketi"] == 0).mean() * 100

    c1.metric("İncelenen Yorum Sayısı", f"{toplam_yorum:,}")
    c2.metric("Ortalama Kullanıcı Puanı", f"{ort_puan:.2f} / 5.00")
    c3.metric("Olumlu Yorum Oranı (4-5★)", f"%{olumlu_oran:.1f}")
    c4.metric("Olumsuz Yorum Oranı (1-2★)", f"%{olumsuz_oran:.1f}")

    st.divider()
    st.subheader("Banka Bazında Karşılaştırmalı Özet Tablo")
    ozet_tablo = df_filtre.groupby(["banka_adi", "banka_turu"]).agg(
        Yorum_Sayisi=("score", "count"),
        Orneklem_Ort_Puan=("score", "mean"),
        Magaza_Genel_Puani=("genel_puan", "first"),
        Indirme_Sayisi=("indirme_sayisi", "first"),
        Ort_Kelime_Sayisi=("kelime_sayisi", "mean")
    ).reset_index()
    ozet_tablo["Orneklem_Ort_Puan"] = ozet_tablo["Orneklem_Ort_Puan"].round(2)
    ozet_tablo["Ort_Kelime_Sayisi"] = ozet_tablo["Ort_Kelime_Sayisi"].round(1)
    st.dataframe(ozet_tablo, use_container_width=True, hide_index=True)

# =====================================================================
# 2. EKRAN: ANALİZ (FİLTRELİ ETKİLEŞİMLİ GRAFİKLER)
# =====================================================================
with tab_analiz:
    col_sol, col_sag = st.columns(2)

    with col_sol:
        st.subheader("Aylık Ortalama Puan Trendi")
        son_aylar = sorted(df_filtre["yorum_ay"].unique())[-6:]
        df_trend = df_filtre[df_filtre["yorum_ay"].isin(son_aylar)].groupby(
            ["yorum_ay", "banka_adi"]
        )["score"].mean().reset_index()

        fig_trend = px.line(
            df_trend, x="yorum_ay", y="score", color="banka_adi", markers=True,
            labels={"yorum_ay": "Ay", "score": "Ortalama Puan", "banka_adi": "Banka"}
        )
        fig_trend.update_yaxes(range=[1, 5.2])
        st.plotly_chart(fig_trend, use_container_width=True)

    with col_sag:
        st.subheader("Şikâyet Konularının Bankalara Göre Dağılımı")
        df_sikayet = df_filtre.dropna(subset=["konu_adi"])
        konu_dagilim = df_sikayet.groupby(["banka_adi", "konu_adi"]).size().reset_index(name="Yorum_Sayisi")

        fig_konu = px.bar(
            konu_dagilim, x="banka_adi", y="Yorum_Sayisi", color="konu_adi", barmode="group",
            labels={"banka_adi": "Banka", "Yorum_Sayisi": "Şikâyet Sayısı", "konu_adi": "Şikâyet Konusu"}
        )
        st.plotly_chart(fig_konu, use_container_width=True)

    st.divider()
    st.subheader("Anonimleştirilmiş Örnek Şikâyet Yorumları (KVKK Uyumlu)")
    secilen_konu_filtre = st.selectbox("İncelemek İstediğiniz Şikâyet Konusunu Seçin:", list(KONU_ISIMLERI.values()))
    ornekler = df_sikayet[df_sikayet["konu_adi"] == secilen_konu_filtre][["banka_adi", "score", "yorum_ay", "content"]].sample(n=min(8, len(df_sikayet[df_sikayet["konu_adi"] == secilen_konu_filtre])), random_state=42)
    st.dataframe(ornekler, use_container_width=True, hide_index=True)

# =====================================================================
# 3. EKRAN: KARAR EKRANI (CANLI DUYGU VE KONU TAHMİNİ)
# =====================================================================
with tab_karar:
    st.subheader("🤖 Müşteri Yorumu Canlı Duygu ve Şikâyet Konusu Tahmini")
    st.write(
        "Aşağıdaki kutuya örnek bir uygulama mağazası yorumu yazın; model duygu durumunu ve şikâyet kategorisini anında tespit etsin.")

    ornek_metin = st.text_area(
        "Müşteri Yorumu Girin:",
        value="Son güncellemeden sonra uygulamaya giriş yapamıyorum, şifre ekranında sürekli atıyor.",
        height=100
    )

    if st.button("Analiz Et ve Tahmin Üret", type="primary"):
        temiz = metin_temizle(ornek_metin)
        if not temiz:
            st.warning("Lütfen anlamlı kelimeler içeren bir yorum girin.")
        else:
            tahmin = duygu_modeli.predict([temiz])[0]
            olasilik = duygu_modeli.predict_proba([temiz])[0]

            c_sonuc1, c_sonuc2 = st.columns(2)
            if tahmin == 1:
                c_sonuc1.success(f"**Duygu Tahmini:** OLUMLU (Güven: %{olasilik[1] * 100:.1f})")
                c_sonuc2.info("**Şikâyet Kategorisi:** Olumlu yorum olduğu için şikâyet konusu atanmadı.")
            else:
                c_sonuc1.error(f"**Duygu Tahmini:** OLUMSUZ (Güven: %{olasilik[0] * 100:.1f})")
                vektor = nmf_tfidf.transform([temiz])
                konu_skorlari = nmf_model.transform(vektor)[0]
                tahmin_konu_id = int(np.argmax(konu_skorlari))
                c_sonuc2.warning(f"**Tespit Edilen Şikâyet Konusu:** {KONU_ISIMLERI[tahmin_konu_id]}")

# =====================================================================
# 4. EKRAN: İMZA ÖZELLİK (YÖNETİM KARAR SİMÜLATÖRÜ)
# =====================================================================
with tab_imza:
    st.subheader("🚀 Yönetim Karar Simülatörü: Hangi Sorun Çözülürse Puan Kaç Olur?")
    st.markdown(
        "Dijital kanallar yöneticisi için **What-If (Ne Olursa?)** analizi: Seçilen bankada belirli bir şikâyet kategorisi çözüldüğünde kazanılan müşterilerin 5 yıldız verdiği varsayımıyla yeni uygulama puanı simüle edilir.")

    col_s1, col_s2, col_s3 = st.columns(3)
    hedef_banka = col_s1.selectbox("Hedef Banka:", df["banka_adi"].unique().tolist())
    hedef_konu = col_s2.selectbox("Çözülecek Öncelikli Sorun:", list(KONU_ISIMLERI.values()))
    iyilestirme_orani = col_s3.slider("Sorun Çözülme Oranı (%):", min_value=10, max_value=100, value=50, step=10)

    df_b = df[df["banka_adi"] == hedef_banka].copy()
    mevcut_puan = df_b["score"].mean()

    # Hedef konudaki olumsuz yorumların puanını 5'e çekme simülasyonu
    konu_maske = (df_b["konu_adi"] == hedef_konu) & (df_b["score"] <= 2)
    etkilenen_yorum_sayisi = int(konu_maske.sum() * (iyilestirme_orani / 100.0))

    mevcut_toplam_puan = df_b["score"].sum()
    mevcut_etkilenen_ort = df_b.loc[konu_maske, "score"].mean() if konu_maske.sum() > 0 else 1.0
    eklenen_puan = etkilenen_yorum_sayisi * (5.0 - mevcut_etkilenen_ort)
    yeni_puan = (mevcut_toplam_puan + eklenen_puan) / len(df_b)
    puan_artisi = yeni_puan - mevcut_puan

    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Mevcut Ortalama Puan", f"{mevcut_puan:.3f}")
    m2.metric("Simüle Edilen Yeni Puan", f"{yeni_puan:.3f}", delta=f"+{puan_artisi:.3f} puan")
    m3.metric("Memnun Edilen Mutsuz Müşteri", f"{etkilenen_yorum_sayisi} kişi")