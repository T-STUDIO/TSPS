# ASTAP インデックス認識・ダウンロード問題の解決ガイド

## 問題の原因分析

### 1. **ファイルアクセス権限の問題**
```
/opt/astap ディレクトリの所有権・パーミッション
- サーバー起動ユーザー（通常は root や特定のサービスユーザー）
- ディレクトリが存在しない、または読み取り専用
```

### 2. **ファイル名の認識ズレ**
```
ダウンロード後：
- ZIP展開時にファイル名が大文字混在で保存される
  例: d80_01.590 vs D80_01.590

検索時：
- 小文字パターン指定では大文字ファイルを検出できない
  例: fnmatch('D80_01.590', 'd80_*.590') → False
```

### 3. **ディレクトリ検索ロジックの不備**
```
元のコード：
- fnmatch()の結果を直接 actual_files に追加
- 複数拡張子パターンの処理が不完全
- ファイル存在判定が曖昧
```

## 解決策

### ステップ1: ディレクトリパーミッション確認

```bash
# サーバー起動ユーザーで実行
ls -la /opt/ | grep astap

# 出力例
drwxr-xr-x  5 root root  4096 Jun 26 10:00 astap

# 問題がある場合は以下で修正
sudo chmod 755 /opt/astap
sudo chmod 644 /opt/astap/*
```

### ステップ2: 正確なファイル認識ロジック

提供するコードの主要な改善：

```python
def check_astap_index_installed(num: str) -> bool:
    """
    大文字小文字を区別せず、複数パターンに対応した検索
    """
    if not os.path.exists(ASTAP_DIR):
        return False
    
    meta = next((x for x in ASTAP_INDEX_METADATA if x["num"] == num), None)
    if not meta:
        return False
    
    try:
        all_files = os.listdir(ASTAP_DIR)  # すべてのファイルを取得
        all_files_lower = [f.lower() for f in all_files]  # 小文字版を生成
        
        expected_patterns = meta.get("expected_files", [])
        
        for pattern in expected_patterns:
            if '*' in pattern:
                # ワイルドカード処理：小文字パターンで検索
                import fnmatch
                if any(fnmatch.fnmatch(f, pattern.lower()) for f in all_files_lower):
                    logger.info(f"Index {num}: Found matching files for pattern {pattern}")
                    return True
            else:
                # 完全一致：大文字小文字を無視
                if pattern.lower() in all_files_lower:
                    logger.info(f"Index {num}: Found exact file {pattern}")
                    return True
        
        logger.warning(f"Index {num}: No files found for patterns {expected_patterns}")
        return False
    
    except Exception as e:
        logger.error(f"Error checking ASTAP index {num}: {e}")
        return False
```

### ステップ3: ZIP展開時の正規化

```python
# 展開時にファイル名を正規化
filename = os.path.basename(member.filename).lower()  # 小文字統一
dest_path = os.path.join(ASTAP_DIR, filename)
```

### ステップ4: デバッグログの活用

```python
# 診断用ログ出力
logger.info(f"ASTAP_DIR: {ASTAP_DIR}")
logger.info(f"Directory exists: {os.path.exists(ASTAP_DIR)}")
logger.info(f"Directory contents: {os.listdir(ASTAP_DIR) if os.path.exists(ASTAP_DIR) else 'N/A'}")
logger.info(f"Expected files for {num}: {meta.get('expected_files', [])}")
```

## テスト手順

### 1. サーバーログで確認

```bash
# サーバーログをリアルタイム監視
tail -f /var/log/solver_server.log | grep -i astap

# 出力例
[INFO] Starting ASTAP index download: D80 from https://www.hnsky.org/d80_v18.zip
[INFO] Extracting d80_v18.zip to /opt/astap...
[INFO] Extracting: d80_v18/d80_01.590 -> /opt/astap/d80_01.590
[INFO] Successfully extracted 18 files for D80
```

### 2. ディレクトリ直接確認

```bash
ls -la /opt/astap/
# d80_01.590, d80_02.590, ... などが見えるはず

# ファイルサイズ確認
du -sh /opt/astap/
```

### 3. WEBUIで再スキャン

- Index Manager にアクセス
- ASTAP タブをクリック
- F5 キーで更新
- チェックボックスが自動的にチェックされていることを確認

## よくある問題と解決策

### 問題A: "ディレクトリが存在せず、作成にも失敗しました"
```
原因: /opt/astap の所有権が別ユーザー
解決:
sudo mkdir -p /opt/astap
sudo chown -R <サーバーユーザー>:<グループ> /opt/astap
sudo chmod 755 /opt/astap
```

### 問題B: "DL成功 → ファイルが見えない"
```
原因: ZIP展開後のファイル名が大文字のまま
解決: 提供コードはこれを自動的に小文字に統一します
```

### 問題C: "特定のインデックス（G05など）しか認識されない"
```
原因: パターンマッチングが拡張子の違いに対応していない
解決: expected_files に複数パターンを指定
      "expected_files": ["g05_*.290", "g05_*.590"]
```

### 問題D: "削除ボタンが機能しない"
```
原因: ファイルが見つからない（認識されていない）
解決: 認識ロジックを修正すれば自動的に解決
```

## メタデータの正確な定義

各インデックスの実際のファイル拡張子（HNSKY公式から確認）:

| インデックス | 説明 | ファイル拡張子 | 視野角 |
|:--|:--|:--|:--|
| D80 | 深空データベース | .590 | 0.15° - 5.0° |
| D50 | 中程度深空 | .290 | 0.8° - 15° |
| V50 | Hipparcos/Tycho | .290 | 0.8° - 15° |
| D20 | 明るい星用 | .290 | 2.0° - 30° |
| D05 | 非常に明るい星用 | .290 | 5.0° - 50° |
| V05 | Hipparcos 明るい星 | .290 | 5.0° - 50° |
| G05 | Gaia DR3 明るい星 | .290 | 5.0° - 50° |
| W08 | 全天スキャン | .290 | 8.0° - 120° |
| hyperleda | 銀河カタログ | .txt, .dat | Any FOV |

## 実装チェックリスト

- [ ] `/opt/astap` ディレクトリのパーミッションを確認
- [ ] 新しい `check_astap_index_installed()` 関数を実装
- [ ] `expected_files` メタデータを全インデックスに追加
- [ ] ZIP展開時に `filename.lower()` を使用
- [ ] ログ出力を詳細化
- [ ] WEB UI で再スキャン機能をテスト
- [ ] ダウンロード・展開・削除を一通りテスト

## トラブルシューティングコマンド

```bash
# ASTAPディレクトリ確認
stat /opt/astap

# ファイル一覧（拡張子別）
find /opt/astap -type f -name "*.590" | head -10
find /opt/astap -type f -name "*.290" | head -10

# サーバープロセスのファイルディスクリプタ確認
lsof | grep /opt/astap

# Python でのテスト
python3 << 'EOF'
import os
import fnmatch

astap_dir = "/opt/astap"
pattern = "d80_*.590"

if os.path.exists(astap_dir):
    files = os.listdir(astap_dir)
    matching = [f for f in files if fnmatch.fnmatch(f.lower(), pattern.lower())]
    print(f"Found {len(matching)} files matching {pattern}")
    print("Files:", matching[:5])
EOF
```

## 追加の最適化

### 非同期スキャン

ディレクトリが大きい場合、非同期スキャンで UI ブロックを防止：

```python
import asyncio

@app.get("/api/scanned_astap_indices")
async def api_scanned_astap_indices():
    # 重い処理を非同期で実行
    await asyncio.sleep(0)
    # ... 以下処理
```

### キャッシング

スキャン結果をキャッシュして処理を高速化：

```python
from datetime import datetime, timedelta

ASTAP_SCAN_CACHE = {"timestamp": None, "data": None}
ASTAP_CACHE_TTL = 60  # 60秒キャッシュ

def get_cached_scan():
    now = datetime.now()
    if (ASTAP_SCAN_CACHE["timestamp"] and 
        now - ASTAP_SCAN_CACHE["timestamp"] < timedelta(seconds=ASTAP_CACHE_TTL)):
        return ASTAP_SCAN_CACHE["data"]
    return None
```

---

このガイドに沿って実装すれば、ASTAP インデックス管理は完全に機能するようになります。
