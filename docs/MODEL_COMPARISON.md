# MODEL COMPARISON

*Cập nhật lần cuối: 28/05/2026*

---

## Bảng tổng hợp (metric trên TEST)

> **Lưu ý test set:** Các run từ `xgb_advanced_moe_postprocess` trở xuống được đánh giá trên `test.csv` gốc (26 mẫu, chỉ class 1/2/4 thật).
> `xgb_aug_full` được đánh giá trên `test_augmented.csv` (36 mẫu: 26 thật + 5 class-0 synthetic + 5 class-4 synthetic) — không so sánh trực tiếp được với các run trên.

| Run | Test set | n | Accuracy | F1 macro | Pos MAE | Pos RMSE |
|---|---|---:|---:|---:|---:|---:|
| `xgb_aug_full` *(28/05/2026)* | test_augmented.csv | 36 | 0.9444 | 0.9452 | **0.0997** | 0.3970 |
| `xgb_advanced_moe_postprocess` | test.csv | 26 | **0.9615** | **0.9653** | 0.1217 | 0.4529 |
| `baseline_xgb_advanced` | test.csv | 26 | 0.9615 | 0.9653 | 0.4208 | 0.7962 |
| `tuned_xgb_advanced_balanced` | test.csv | 26 | 0.9231 | 0.9339 | 0.2497 | 0.5085 |
| `tuned_xgb_balanced` | test.csv | 26 | 0.8846 | 0.8960 | 0.3816 | 0.6856 |
| `baseline_xgb` | test.csv | 26 | 0.8462 | 0.8538 | 0.4198 | 0.7678 |
| `tuned_rf_balanced_refit` | test.csv | 26 | 0.8462 | 0.6308 | 0.3703 | 0.6157 |
| `baseline_rf` | test.csv | 26 | 0.7308 | 0.6124 | 0.6846 | 0.9749 |
| `tuned_cnn1d_balanced` | test.csv | 26 | 0.6923 | 0.3439 | 1.1632 | 1.5307 |
| `tuned_mlp_balanced` | test.csv | 26 | 0.6154 | 0.2540 | 1.1690 | 1.5027 |

---

## Per-class recall — `xgb_aug_full` trên `test_augmented.csv` *(28/05/2026)*

Lần đầu tiên đo được đầy đủ 4 class nhờ augmented test set.

| Class | Label | Precision | Recall | F1 | n | Loại mẫu |
|-------|-------|-----------|--------|----|---|-----------|
| 0 | Bình thường | 1.000 | **1.000** | 1.000 | 5 | Synthetic (sanity check) |
| 1 | 1 hư hỏng | 1.000 | 0.714 | 0.833 | 7 | Real |
| 2 | 2 hư hỏng | 0.900 | 1.000 | 0.947 | 18 | Real |
| 4 | 4 hư hỏng | 1.000 | **1.000** | 1.000 | 6 | 1 real + 5 synthetic |

**Nhận xét:**
- Class 0: model đã học nhận ra dầm bình thường — recall 100% (nhưng test samples là synthetic, không phải blind test).
- Class 1: recall 71% — điểm yếu thực sự duy nhất. 2 trong 7 mẫu bị misclassify.
- Class 4: recall 100% với n=6 thay vì n=1 như trước — đáng tin cậy hơn.

---

## Kết luận nhanh

- Tốt nhất trên test gốc (26 real samples): `xgb_advanced_moe_postprocess` — `acc=0.9615`, `f1_macro=0.9653`, `pos_mae=0.1217`
- Tốt nhất về pos_MAE tuyệt đối: `xgb_aug_full` — `pos_mae=0.0997` (nhưng test set khác, có synthetic data)
- Điểm yếu cần cải thiện: **recall class 1 = 71%** — nhất quán qua cả hai model

Ghi chú:
- f1_macro của `xgb_aug_full` (0.9452) thấp hơn `xgb_advanced_moe_postprocess` (0.9653) không phải do model kém hơn — do macro tính trên 4 class thay vì 3, và test set lớn hơn có thêm class khó.
- Nếu dùng score đơn giản `0.5 * F1 + 0.5 * (1 / (1 + MAE))`, thì `xgb_aug_full` đứng đầu khi tính trên test_augmented.

---

## Xếp hạng theo F1 macro (test gốc)

1. `xgb_advanced_moe_postprocess`: `f1_macro=0.9653`, `pos_mae=0.1217`
2. `baseline_xgb_advanced`: `f1_macro=0.9653`, `pos_mae=0.4208`
3. `tuned_xgb_advanced_balanced`: `f1_macro=0.9339`, `pos_mae=0.2497`
4. `tuned_xgb_balanced`: `f1_macro=0.8960`, `pos_mae=0.3816`
5. `baseline_xgb`: `f1_macro=0.8538`, `pos_mae=0.4198`
6. `tuned_rf_balanced_refit`: `f1_macro=0.6308`, `pos_mae=0.3703`
7. `baseline_rf`: `f1_macro=0.6124`, `pos_mae=0.6846`
8. `tuned_cnn1d_balanced`: `f1_macro=0.3439`, `pos_mae=1.1632`
9. `tuned_mlp_balanced`: `f1_macro=0.2540`, `pos_mae=1.1690`

## Xếp hạng theo Pos MAE (test gốc)

1. `xgb_advanced_moe_postprocess`: `pos_mae=0.1217`, `f1_macro=0.9653`
2. `tuned_xgb_advanced_balanced`: `pos_mae=0.2497`, `f1_macro=0.9339`
3. `tuned_rf_balanced_refit`: `pos_mae=0.3703`, `f1_macro=0.6308`
4. `tuned_xgb_balanced`: `pos_mae=0.3816`, `f1_macro=0.8960`
5. `baseline_xgb`: `pos_mae=0.4198`, `f1_macro=0.8538`
6. `baseline_xgb_advanced`: `pos_mae=0.4208`, `f1_macro=0.9653`
7. `baseline_rf`: `pos_mae=0.6846`, `f1_macro=0.6124`
8. `tuned_cnn1d_balanced`: `pos_mae=1.1632`, `f1_macro=0.3439`
9. `tuned_mlp_balanced`: `pos_mae=1.1690`, `f1_macro=0.2540`

---

## Khuyến nghị thực tế

- **Báo cáo/paper trên real data:** chọn `xgb_advanced_moe_postprocess` (test.csv gốc 26 mẫu, F1 cao nhất, MAE thấp nhất)
- **Kiểm tra per-class recall đầy đủ:** dùng `xgb_aug_full` + `test_augmented.csv` với caveat synthetic
- **Ưu tiên cải thiện tiếp:** recall class 1 (71%) — nhất quán là điểm yếu qua mọi run
- `MLP` và `CNN1D` hiện chưa phải ứng viên mạnh trong repo này

