# Datasets (Download & Paths)

> **Important:** Please download datasets manually (via Kaggle or official sources) and update the YAML config paths accordingly.

---

## 5G-NIDD (5G Non-IP Data Delivery)
- Kaggle: [5G-NIDD Dataset](https://www.kaggle.com/datasets/humera11/5g-nidd-dataset)

**Expected file (example used in configs):**
- Unified CSV (e.g., `Combined.csv`):
  ```yaml
  data:
    csv_path: "/path/to/Combined.csv"
  ```

---

## InSDN (SDN Intrusion Dataset)
- Kaggle: [InSDN Dataset](https://www.kaggle.com/datasets/badcodebuilder/insdn-dataset)

**Expected structure (as in `configs/insdn.yaml`):**
```yaml
data:
  dataset: "insdn"
  base_dir: "/path/to/InSDN_DatasetCSV"
  files:
    metasploitable-2.csv: 1
    Normal_data.csv: 0
    OVS.csv: 0
```

---

## CICIDS2017
- Official page: [CICIDS2017 — Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html)

**Usage with this repo:**
- Either point to a **single CSV** (e.g., PortScan subset), or to a **folder** with multiple CSVs:
  ```yaml
  data:
    dataset: "cicids"
    path: "/path/to/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"  # or a folder with *.csv
  ```
