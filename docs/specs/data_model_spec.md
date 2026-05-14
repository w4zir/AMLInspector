# AML Detection System Specification
**Project:** Wise-Style Localized Entity Monitoring (IBM Dataset Implementation)

| Metadata Field | Value |
| :--- | :--- |
| **Document Status** | Draft / Technical Specification |
| **Framework** | Champion-Challenger Architecture |
| **Data Source** | IBM Transactions for AML (Synthetic) |
| **Primary Actor** | Home Bank (Single-Org Visibility) |

---

## 1. Project Objective
To design and implement an Anti-Money Laundering (AML) detection pipeline that simulates the operational reality of a FinTech (e.g., Wise). The system must detect suspicious patterns using **only** internal transaction logs, locally derived entity relationships, and public risk indicators, effectively handling the "half-blind" nature of cross-border financial monitoring.

---

## 2. Data Preprocessing & Partitioning
The IBM dataset contains global visibility. To simulate a "Wise" environment, we must enforce **data silos** during the preprocessing phase.

### 2.1 The "Home Bank" Filter
* **Bank Selection:** Designate a high-volume Bank ID (e.g., *Bank 789*) as the **Home Bank**.
* **Visibility Masking:**
    * Full details (Account ID, History, Balance) are kept for all transactions where `From Bank` or `To Bank` is the Home Bank.
    * For external legs, only the `Bank ID` and `Currency` are treated as features; the specific `Account ID` of the external party is hashed to prevent "illegal" data leakage during training.

### 2.2 Cleaning & Normalization
* **Temporal Alignment:** Convert timestamps to UTC.
* **Currency Standardization:** Since Wise deals with multi-currency, all transaction amounts are converted to a base currency (USD) using a fixed or historical exchange rate lookup to enable *Amount*-based feature comparison.

---

## 3. Feature Engineering Layer
Features are engineered across four distinct domains to capture the complexity of "Layering" and "Integration."

### A. Transactional & Corridor Features
* **Velocity ($V_{24h}$):** Number of outbound transactions in the last 24 hours.
* **Corridor Risk Score:** A weight assigned to the `Source_Country` → `Target_Country` route (e.g., UK to Cayman Islands).
* **Dwell Time:** Time elapsed between an inbound transfer and an outbound transfer for the same internal account. $\Delta t = t_{out} - t_{in}$.
* **Amount Roundness:** Binary flag if $Amount \pmod{100} = 0$.

### B. Entity & Community Features
* **Internal Fan-Out:** Ratio of unique internal recipients to total transaction volume.
* **Shared PII Linkage:** Number of accounts sharing the same device fingerprint or IP address (Internal Entity Resolution).
* **Local Clustering Coefficient:** Measures how "interconnected" a group of internal accounts is. High clustering among unrelated names often signals a money mule ring.

### C. Publicly Available & External Data
* **HRJ Status:** Boolean flag if the counterparty bank is located in a High-Risk Jurisdiction (FATF Grey/Black list).
* **Company Age Proxy:** For business accounts, the number of days since the first observed transaction in the system. *(AML Red Flag: High-volume activity from a company < 90 days old).*
* **PEP Match:** Mock indicator of Politically Exposed Person status for the account holder.

---

## 4. Modeling Architecture: Champion-Challenger
To ensure both reliability and adaptability, the system employs a dual-model approach.

### 4.1 The Champion Model (Supervised)
* **Algorithm:** XGBoost (Gradient Boosted Decision Trees).
* **Target:** Predicting the `Is_Laundering` label.
* **Logic:** Trained on historical SARs. It excels at identifying **Known Knowns** (e.g., Smurfing patterns, classic structuring).
* **Optimization:** Uses `Scale_Pos_Weight` to handle the 1:10,000 class imbalance.

### 4.2 The Challenger Model (Unsupervised)
* **Algorithm:** Isolation Forest or Variational Autoencoder (VAE).
* **Logic:** Ignores labels. It identifies **Unknown Unknowns** by flagging transactions that occupy low-density regions in the feature space.
* **Role:** If the Challenger flags a transaction that the Champion missed, it is sent for high-priority manual review to identify evolving laundering typologies.

---

## 5. Implementation Roadmap
1.  **Phase 1:** Partition IBM dataset and build the "Home Bank" views.
2.  **Phase 2:** Develop the feature engineering pipeline using Python/Pandas and NetworkX for local graphs.
3.  **Phase 3:** Train XGBoost (Champion) on labeled data.
4.  **Phase 4:** Deploy Isolation Forest (Challenger) to detect anomalies.
5.  **Phase 5:** Evaluation using a Confusion Matrix focused on the "Cost of Missed Detection" vs "Cost of Investigation."
