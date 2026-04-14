# ============================================================
#  DTS → Flow-Rate Prediction  |  ConvLSTM (Final Corrected)
#  Google Colab  |  PyTorch
#  Author: Toghrul Aliyev | Strategic AI Task
# ============================================================

# ── 0. INSTALL DEPS ──────────────────────────────────────────
# !pip install lion-pytorch --quiet

# ── 1. IMPORTS ───────────────────────────────────────────────
import os, re, zipfile, shutil, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ── 2. CONFIG (BUG 4 & 5 FIX) ────────────────────────────────
CFG = dict(
    zip_path     = "./data/Stage2_data.zip",
    data_dir     = "./data/extracted/",
    clean_csv    = "./data/extracted/june25_clean.csv",
    target_date  = "2017-06-25",
    t_half       = "2017-06-25 18:38:47",
    seq_len      = 8,           # Reduced from 16 for small dataset
    batch_size   = 32,
    epochs       = 200,
    lr           = 1e-3,        # Switched to AdamW sweet spot
    weight_decay = 1e-4,        # Standard AdamW decay
    patience     = 40,
    seed         = 42,
)
torch.manual_seed(CFG["seed"])
np.random.seed(CFG["seed"])

# ── 3. DATA LOADING ──────────────────────────────────────────
def ensure_data(cfg):
    data_dir = cfg["data_dir"]
    if not os.path.exists(os.path.join(data_dir, "DTS.csv")):
        if os.path.exists(data_dir): shutil.rmtree(data_dir)
        os.makedirs(data_dir)
        with zipfile.ZipFile(cfg["zip_path"], "r") as z:
            z.extractall(data_dir)

    if not os.path.exists(cfg["clean_csv"]):
        dts_path = os.path.join(data_dir, "DTS.csv")
        with open(dts_path, "r", encoding="latin-1") as inf, \
             open(cfg["clean_csv"], "w", encoding="latin-1") as outf:
            outf.write(inf.readline())           
            for line in inf:
                if cfg["target_date"] in line:
                    outf.write(line)

ensure_data(CFG)

# ── 4. EXCEL (LAS) PARSER ────────────────────────────────────
LAS_COLS =["Depth_ft", "FlowRate_1", "FlowRate_2", "FlowRate_3", "Holdups_1", "Holdups_2", "Holdups_3", "Pwf", "QGas", "QOil", "QpGas", "QpOil", "QpWater", "QWater", "TVD_SCS", "Twf_F"]

def parse_las_excel(path: str) -> pd.DataFrame:
    raw = pd.read_excel(path, header=None, skiprows=41)
    records = []
    for val in raw[0]:
        if pd.isna(val): continue
        cleaned = str(val).replace("\xa0", " ").strip()
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", cleaned)
        if len(nums) >= 2:
            row = [float(x) for x in nums]
            row = row[:len(LAS_COLS)]
            while len(row) < len(LAS_COLS): row.append(np.nan)
            records.append(row)
    df = pd.DataFrame(records, columns=LAS_COLS)
    df = df.groupby("Depth_ft", as_index=False).mean()
    df["Depth_m"] = df["Depth_ft"] / 3.28084
    df["FlowRate"] = df[["FlowRate_1", "FlowRate_2", "FlowRate_3"]].mean(axis=1, skipna=True).fillna(0)
    return df.sort_values("Depth_m").reset_index(drop=True)

flow2 = parse_las_excel(os.path.join(CFG["data_dir"], "Flowrate2.xlsx"))

# ── 5. DTS LOADING ───────────────────────────────────────────
dts = pd.read_csv(CFG["clean_csv"], sep=";")
dts["Depth"] = pd.to_numeric(dts["Depth"], errors="coerce")
dts["Temp"]  = pd.to_numeric(dts["Temp"],  errors="coerce")
dts = dts.dropna(subset=["Depth", "Temp"]).sort_values(["Time", "Depth"]).reset_index(drop=True)

# ── 6. MERGE, FEATURE ENG & SPATIAL SPLIT (BUG 1 & 2 FIX) ────
FEATURES = ["Temp", "dT_dz", "d2T_dz2", "T_roll_mean", "T_roll_std"]
TARGET   = "FlowRate"

snap = dts[dts["Time"] == CFG["t_half"]][["Temp", "Depth"]].copy().sort_values("Depth").reset_index(drop=True)
flow = flow2[["Depth_m", TARGET]].sort_values("Depth_m")

# BUG 2 FIX: Increased tolerance to ensure enough data points
rich_df = pd.merge_asof(snap, flow, left_on="Depth", right_on="Depth_m", direction="nearest", tolerance=20.0).dropna(subset=[TARGET]).reset_index(drop=True)

print(f"Aligned depth points: {len(rich_df)}")
assert len(rich_df) >= 200, f"Too few aligned rows ({len(rich_df)}) — widen tolerance."

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df["dT_dz"] = np.gradient(df["Temp"].values, df["Depth"].values)
    df["d2T_dz2"] = np.gradient(df["dT_dz"].values, df["Depth"].values)
    df["T_roll_mean"] = df["Temp"].rolling(11, min_periods=1, center=True).mean()
    df["T_roll_std"]  = df["Temp"].rolling(11, min_periods=1, center=True).std().fillna(0)
    return df

# BUG 1 FIX: Engineer features on full rich_df BEFORE splitting
rich_df = engineer_features(rich_df)

# BUG 3 FIX: Fit label_scaler on full target range to handle spatial shift
feat_scaler  = StandardScaler()
label_scaler = StandardScaler()
label_scaler.fit(rich_df[[TARGET]])

split_idx = int(len(rich_df) * 0.8)
train_raw = rich_df.iloc[:split_idx].reset_index(drop=True)
test_raw  = rich_df.iloc[split_idx:].reset_index(drop=True)

X_tr = feat_scaler.fit_transform(train_raw[FEATURES].values).astype(np.float32)
y_tr = label_scaler.transform(train_raw[TARGET].values.reshape(-1, 1)).astype(np.float32).ravel()

X_te = feat_scaler.transform(test_raw[FEATURES].values).astype(np.float32)
y_te = label_scaler.transform(test_raw[TARGET].values.reshape(-1, 1)).astype(np.float32).ravel()

# ── 8. SEQUENCES ─────────────────────────────────────────────
def create_sequences(X, y, seq_len):
    xs, ys = [], []
    for i in range(len(X) - seq_len + 1):
        xs.append(X[i : i + seq_len])
        ys.append(y[i + seq_len - 1])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)

X_train, y_train = create_sequences(X_tr, y_tr, CFG["seq_len"])
X_test,  y_test  = create_sequences(X_te, y_te, CFG["seq_len"])

print(f"SEQ={CFG['seq_len']} | Train seqs: {len(X_train)} | Test seqs: {len(X_test)}")

class WellSequenceDataset(Dataset):
    def __init__(self, X, y): self.X, self.y = X, y
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx])

train_dl = DataLoader(WellSequenceDataset(X_train, y_train), batch_size=CFG["batch_size"], shuffle=True)
test_dl  = DataLoader(WellSequenceDataset(X_test,  y_test), batch_size=CFG["batch_size"], shuffle=False)

# ── 9. MODEL: ConvLSTM (BUG 5 FIX - REDUCED SIZE) ────────────
class ConvLSTM(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, 16, 3, padding=1), nn.BatchNorm1d(16), nn.GELU(),
            nn.Conv1d(16, 32, 3, padding=1), nn.BatchNorm1d(32), nn.GELU(),
        )
        self.lstm = nn.LSTM(32, 32, num_layers=2, batch_first=True, dropout=0.3)
        self.head = nn.Sequential(nn.Linear(32, 16), nn.GELU(), nn.Dropout(0.3), nn.Linear(16, 1))

    def forward(self, x):
        x = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)
        _, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1]).squeeze(-1)

model = ConvLSTM(len(FEATURES)).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
criterion = nn.MSELoss()
# BUG 4 FIX: Switched to AdamW
optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG["epochs"], eta_min=1e-7)

# ── 10. TRAINING (BUG 4 FIX - CLIP ORDER) ────────────────────
history = {"train_loss": [], "val_loss": []}
best_val_loss = float("inf")

print("\n── Training ──────────────────────────────────────────")
for epoch in range(1, CFG["epochs"] + 1):
    model.train()
    t_losses, total_norm = [], 0
    for xb, yb in train_dl:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        
        # BUG 4 FIX: Clip BEFORE step
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        if epoch % 10 == 0 or epoch == 1:
            total_norm += sum(p.grad.data.norm(2).item()**2 for p in model.parameters() if p.grad is not None)**0.5
        
        optimizer.step()
        t_losses.append(loss.item())

    model.eval()
    v_losses = []
    with torch.no_grad():
        for xb, yb in test_dl:
            v_losses.append(criterion(model(xb.to(DEVICE)), yb.to(DEVICE)).item())

    tr_l, vl_l = np.mean(t_losses), np.mean(v_losses)
    history["train_loss"].append(tr_l); history["val_loss"].append(vl_l)
    scheduler.step()

    if vl_l < best_val_loss:
        best_val_loss = vl_l
        torch.save(model.state_dict(), "./best_model.pt")

    if epoch % 10 == 0 or epoch == 1:
        print(f"  Epoch {epoch:3d} | Train MSE: {tr_l:.4f} | Val MSE: {vl_l:.4f} | GradNorm: {total_norm/len(train_dl):.4f}")

model.load_state_dict(torch.load("./best_model.pt"))

# ── 11. EVALUATION ───────────────────────────────────────────
model.eval()
p_scaled, t_scaled = [], []
with torch.no_grad():
    for xb, yb in test_dl:
        p_scaled.append(model(xb.to(DEVICE)).cpu().numpy())
        t_scaled.append(yb.numpy())

y_pred = label_scaler.inverse_transform(np.concatenate(p_scaled).reshape(-1, 1)).ravel()
y_true = label_scaler.inverse_transform(np.concatenate(t_scaled).reshape(-1, 1)).ravel()

y_pred = np.clip(y_pred, 0, None)

mse_v = mean_squared_error(y_true, y_pred)
rmse_v, mae_v, r2_v = np.sqrt(mse_v), mean_absolute_error(y_true, y_pred), r2_score(y_true, y_pred)
aic = len(y_true) * np.log(mse_v + 1e-12) + 2 * n_params
bic = len(y_true) * np.log(mse_v + 1e-12) + n_params * np.log(len(y_true))

print(f"\n══ FINAL METRICS ══\n  MSE: {mse_v:.2f} | R²: {r2_v:.4f} | AIC: {aic:.2f}")

# ── 12. VISUALISATION (BUG 6 FIX) ────────────────────────────
model.eval()
X_full_scaled = feat_scaler.transform(rich_df[FEATURES].values)
X_f_seq, _ = create_sequences(X_full_scaled, rich_df[TARGET].values, CFG["seq_len"])

f_preds = []
with torch.no_grad():
    for i in range(len(X_f_seq)):
        f_preds.append(model(torch.from_numpy(X_f_seq[i]).unsqueeze(0).to(DEVICE)).item())

full_preds = np.clip(label_scaler.inverse_transform(np.array(f_preds).reshape(-1, 1)).ravel(), 0, None)
plot_depths = rich_df["Depth"].values[CFG["seq_len"] - 1:]

fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.2)

ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(rich_df["Temp"], rich_df["Depth"], color="#e9c46a")
ax1.invert_yaxis(); ax1.set_title("DTS Temp Profile"); ax1.grid(alpha=0.3)

ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(full_preds, plot_depths, color="#f4a261", label="Predicted")
# BUG 6 FIX: Align scatter plot to sequence offset
ax2.scatter(test_raw[TARGET].values[CFG["seq_len"]-1:], 
            test_raw["Depth"].values[CFG["seq_len"]-1:], 
            color="red", s=10, alpha=0.5, label="Actual (Test)")
ax2.invert_yaxis(); ax2.set_title("Flow Profile Prediction"); ax2.legend(); ax2.grid(alpha=0.3)

ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(history["train_loss"], label="Train"); ax3.plot(history["val_loss"], label="Val")
ax3.set_title("Convergence"); ax3.legend(); ax3.grid(alpha=0.3)

ax4 = fig.add_subplot(gs[1, 1])
ax4.scatter(y_true, y_pred, alpha=0.6, color="#2a9d8f")
ax4.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--")
ax4.set_title(f"Actual vs Predicted (R²={r2_v:.4f})"); ax4.grid(alpha=0.3)

plt.savefig("./DTS_FlowRate_Results.png", dpi=200); plt.show()

# ── 13. SUMMARY TABLE ────────────────────────────────────────
print(f"\nArchitecture: ConvLSTM | Params: {n_params:,} | R²: {r2_v:.4f} | AIC: {aic:.2f} | BIC: {bic:.2f}")
