# パフォーマンス計測

## 計測内容

ローカル検証スクリプトで `SAM -> MatAnyone -> 動画出力` の一連の処理を計測しました。

```powershell
.\.venv\Scripts\python.exe scripts\run_pipeline_check.py --input media\bookcat.mp4 --device cpu --performance_profile fast --cpu_threads 8 --sam_model_type vit_h --frame_limit 241 --positive_point 280,180 --negative_point 30,30 --negative_point 530,30 --output_dir results\bookcat-profile-exp\fast
```

使用した入力条件:

- 入力: `media/bookcat.mp4`
- 解像度: `560x560`
- フレーム数: `241`
- クリック位置: 正例 `280,180`、負例 `30,30` と `530,30`

生成物は次の場所に保存されています。

- `results/bookcat-profile-exp/`
- `results/bookcat-profile-exp-gpu/`
- `results/bookcat-profile-exp-compare.json`
- `media/bookcat-quality-preview.webp`
- `media/bookcat-balanced-preview.webp`
- `media/bookcat-fast-preview.webp`

見比べ用の代表フレーム:

- `results/bookcat-profile-exp/comparison_frame_120.png`

軽量プレビュー:

| `quality` | `balanced` | `fast` |
| --- | --- | --- |
| <img src="/bookcat-quality-preview.webp" alt="Quality profile preview" width="180"> | <img src="/bookcat-balanced-preview.webp" alt="Balanced profile preview" width="180"> | <img src="/bookcat-fast-preview.webp" alt="Fast profile preview" width="180"> |

## 速度比較

以下の時間は、マスク生成・matting・動画保存まで含んだ合計時間です。

| プロファイル | CPU時間 | GPU時間 | GPU高速化率 |
| --- | ---: | ---: | ---: |
| `quality` | `227.76s` | `72.84s` | `3.13x` |
| `balanced` | `225.62s` | `78.72s` | `2.87x` |
| `fast` | `211.33s` | `71.43s` | `2.96x` |

## 品質差分

`quality` を基準出力として比較しました。

| プロファイル | Foreground MAE | Alpha MAE | Foreground PSNR | Alpha PSNR |
| --- | ---: | ---: | ---: | ---: |
| `balanced` | `0.87` | `0.63` | `33.82 dB` | `30.31 dB` |
| `fast` | `1.48` | `1.92` | `29.04 dB` | `24.14 dB` |

## まとめ

- この `560x560` 動画では、`balanced` は `quality` にかなり近い見た目でした。
- `fast` が最速ですが、入力が小さいため差は限定的でした。
- 今回の条件では、GPU はアプリ全体でおおむね CPU の `3x` 前後でした。
- 品質差は、合成済み foreground より先に alpha matte に表れやすいです。

## 再実行コマンド

CPU:

```powershell
.\.venv\Scripts\python.exe scripts\run_pipeline_check.py --input media\bookcat.mp4 --device cpu --performance_profile balanced --cpu_threads 8 --sam_model_type vit_h --frame_limit 241 --positive_point 280,180 --negative_point 30,30 --negative_point 530,30 --output_dir results\bookcat-profile-exp\balanced
```

GPU:

```powershell
.\.venv\Scripts\python.exe scripts\run_pipeline_check.py --input media\bookcat.mp4 --device cuda --performance_profile balanced --sam_model_type vit_h --frame_limit 241 --positive_point 280,180 --negative_point 30,30 --negative_point 530,30 --output_dir results\bookcat-profile-exp-gpu\balanced
```

Animated WebP 生成:

```powershell
uv run --project D:\Prj\video-background-remover video-background-remover D:\Prj\MatAnyone\results\bookcat-profile-exp\balanced\bookcat_foreground.mp4 D:\Prj\MatAnyone\media\bookcat-balanced-preview.webp --matanyone --alpha-video D:\Prj\MatAnyone\results\bookcat-profile-exp\balanced\bookcat_alpha.mp4 --animated webp --webp-fps 6 --max-frames 96 --size 280x280
```

`--size 280x280` が、元の `560x560` サンプルに対する半分サイズの後処理です。
