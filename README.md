# 🧠 Brain Tumor Detector (SegResNet 3D)

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/By-djuro/Brain-tumor-detector)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![MONAI](https://img.shields.io/badge/MONAI-SegResNet-28A745)](https://monai.io/)
[![Gradio](https://img.shields.io/badge/Gradio-Web%20UI-FF7C00)](https://www.gradio.app/)

3D multimodalna segmentacija tumora mozga iz MRI snimki korištenjem **MONAI `SegResNet`** arhitekture trenirane na **BraTS 2020** skupu podataka. Projekt sadrži web aplikaciju u **Gradio** sučelju (hostanu na Hugging Face Spaces) te lokalni desktop preglednik izrađen u **PyQt6**.

---

## ✨ Značajke

* **Multimodalni 3D ulaz:** Prima 4 standardna MRI modaliteta (`T1`, `T1ce`, `T2`, `FLAIR`) u NIfTI formatu (`.nii` / `.nii.gz`).
* **Automatsko poravnanje i predobrada:** Kanonska orijentacija volumena (`nibabel.as_closest_canonical`), provjera dimenzija i afine matrice te Z-score normalizacija na ne-nulitim vokselima.
* **Sliding Window 3D inferencija:** Evaluacija volumena s preklapanjem (`overlap=0.5`, Gaussian weighting) i mješovitom preciznošću (`FP16` na CUDA GPU-u).
* **Interaktivni 2×2 MRI preglednik:** Brzo listanje po Z-slojevima, kontrola prozirnosti maske i uključivanje/isključivanje prikaza segmentacije.
* **Izvoz rezultata:** Preuzimanje generirane 3D maske segmentacije u `_seg.nii` formatu s očuvanom afinom matricom.

---

## 🧬 Klase segmentacije (BraTS standard)

Model segmentira tri tumorske subregije:

| Oznaka | Boja u prikazu | Naziv klase | Opis |
| :---: | :---: | :--- | :--- |
| **1** | 🔴 Crvena | **NCR / NET** | Nekrotična jezgra i nepojačavajući tumor (*Necrotic and Non-Enhancing Tumor Core*) |
| **2** | 🟢 Zelena | **ED** | Peritumoralni edem (*Peritumoral Edema*) |
| **4** | 🔵 Plava | **ET** | Aktivni tumor koji prima kontrast (*Enhancing Tumor*) |

*(Napomena: Interna izlazna klasa `3` modela automatski se mapira u BraTS oznaku `4`.)*

---

## 🏗️ Arhitektura i konfiguracija modela

* **Mreža:** `SegResNet` (`spatial_dims=3`, `init_filters=16`, `in_channels=4`, `out_channels=4`, `dropout_prob=0.2`)
* **Veličina patcha (`ROI`):** `(64, 64, 64)`
* **Težine modela:** `best_segresnet.pt`

---

## 🚀 Korištenje aplikacije

### 1. Online putem Hugging Face Spaces
Aplikaciji možeš pristupiti izravno u pregledniku bez instalacije:
1. Otvori aplikaciju na linku: https://huggingface.co/spaces/djuroslav/brain-tumor-detector_by-djuro
2. Učitaj sva 4 NIfTI modaliteta za istog pacijenta (`*_t1.nii`, `*_t1ce.nii`, `*_t2.nii`, `*_flair.nii`).
3. Klikni **▶ Pokreni segmentaciju**.
4. Pregledaj rezultate pomoću klizača za **Z-sloj** i **prozirnost**, ili preuzmi `_seg.nii` datoteku.

### 2. Lokalno pokretanje (Gradio Web App)

```bash
# Kloniraj repozitorij
git clone https://github.com/<tvoje-korisnicko-ime>/<naziv-repozitorija>.git
cd <naziv-repozitorija>

# Instaliraj ovisnosti
pip install -r requirements.txt

# Pokreni Gradio aplikaciju
python app.py
```

### 3. Lokalni desktop preglednik (PyQt6)
Za lokalni pregled NIfTI volumena i maski bez web preglednika pokreni desktop skriptu:

```bash
python lokalni_nifti_viewer.py
```

---

## 📁 Struktura repozitorija

```text
├── app.py                   # Gradio web aplikacija za 3D segmentaciju i vizualizaciju
├── lokalna_inferencija.py   # Skripta za lokalnu segmentaciju novih MRI (NIfTI) snimki
├── lokalni_nifti_viewer.py  # Lokalni PyQt6 desktop preglednik za NIfTI snimke
├── best_segresnet.pt        # Trenirane težine SegResNet modela
├── requirements.txt         # Python paketi potrebni za pokretanje
└── README.md                # Dokumentacija projekta
```

---

## ⚠️ Napomena

Ova aplikacija razvijena je isključivo u istraživačke i edukativne svrhe te **nije namijenjena za kliničku dijagnostiku** ili donošenje medicinskih odluka.

## 📄 Licenca

Projekt je objavljen pod [MIT licencom](LICENSE).
