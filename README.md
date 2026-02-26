# goda-wb

合田の砕波変形モデル（Goda's breaking wave deformation model）を実装したPythonパッケージです。

## 概要

このパッケージは、合田の砕波変形モデルを用いて、浅水域における波の変形を計算します。代表波高（H1/1000, H1/250, H1/100, H1/50, H1/10, H1/5, H1/3）や浅水係数などを算出できます。

## インストール

```bash
```

## 使用方法

### 基本的な使用例

```python
import goda_wb.core as core

# 海底勾配、波形勾配、水深波長比を指定して計算
tant = 0.01  # 海底勾配
h0l0 = 0.02  # 波形勾配（H0/L0）
dl0 = [0.4, 0.3, 0.2]  # 水深波長比（d/L0）のリスト

df = core.cal_surf_goda(tant, h0l0, dl0)
print(df)
```

### 有次元での計算

```python
# 沖波波高、周期、水深を指定して計算
H0 = 12.0  # 沖波波高 [m]
T = 12.0   # 周期 [s]
tant = 0.01  # 海底勾配

# デフォルトの水深波長比リストを使用
df = core.cal_surf_goda_dim(tant=tant, H0=H0, T=T, dim=True)

# 特定の水深を指定
d = 14.0  # 水深 [m]
df = core.cal_surf_goda_dim(tant=tant, H0=H0, T=T, d=d, dim=True)
```

### 単一点の計算

```python
# 単一の水深波長比での計算
df = core.cal_surf_goda_point(tant=0.01, h0l0=0.02, dl0=0.3)
```

### 浅水係数の計算

```python
# 浅水係数の計算
dl0 = 0.3  # 水深波長比
h0l0 = 0.02  # 波形勾配
aks = core.shoal(dl0, h0l0)
print(f"浅水係数: {aks}")
```

### 波長の計算

```python
# 波長の計算
d = 10.0  # 水深 [m]
T = 12.0  # 周期 [s]
L = core.cal_wave_length(d, T)
print(f"波長: {L} [m]")
```

## 主な機能

### `cal_surf_goda(tant, h0l0, dl0)`

合田の砕波変形モデルによる計算を実行します。

**パラメータ:**
- `tant`: 海底勾配
- `h0l0`: 波形勾配（H0/L0）
- `dl0`: 水深波長比（d/L0）のリスト、またはNone（デフォルト値を使用）

**戻り値:**
- `pandas.DataFrame`: 以下のカラムを含むデータフレーム
  - `H1_1000`, `H1_250`, `H1_100`, `H1_50`, `H1_10`, `H1_5`, `H1_3`: 各代表波高比
  - `dh0`: 水深波高比（h/H0）
  - `dl0`: 水深波長比（d/L0）
  - `etal0`: セットアップ高さ比
  - `aks`: 浅水係数

### `cal_surf_goda_dim(tant, H0, T, dim=False, d=None)`

有次元での計算を実行します。

**パラメータ:**
- `tant`: 海底勾配
- `H0`: 沖波波高 [m]
- `T`: 周期 [s]
- `dim`: Trueの場合、有次元の結果も含める
- `d`: 水深 [m]（単一値、リスト、またはNone）

### `shoal(dl0, h0l0)`

浅水変形の計算を行い、浅水係数を返します。

**パラメータ:**
- `dl0`: 水深波長比
- `h0l0`: 波形勾配

**戻り値:**
- `float`: 浅水係数

### `cal_wave_length(d, T)`

波長を計算します。

**パラメータ:**
- `d`: 水深 [m]
- `T`: 周期 [s]

**戻り値:**
- `float`: 波長 [m]

## 依存関係

- Python >= 3.11
- numpy >= 2.3.4
- pandas >= 2.3.3
- scipy >= 1.16.3

## 開発

開発環境のセットアップ:

```bash
# 依存関係のインストール
#pip install -e ".[dev]"
```

テストの実行:

```bash
pytest
```

## ライセンス


## 参考文献

- 合田良実, "浅海域における波浪の砕波変形", 港湾空港技術研究所報告, Vol. 14, No. 3, 1975. ([全文PDF](https://www.pari.go.jp/PDF/vol014-no03-03.pdf) / [書誌情報](https://www.pari.go.jp/1975/09/1975090140303.html))
