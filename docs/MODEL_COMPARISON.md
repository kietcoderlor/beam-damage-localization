# MODEL COMPARISON

## Bảng tổng hợp (metric trên TEST)

| Run | Nhóm model | Accuracy | F1 macro | Pos MAE | Pos RMSE |
|---|---|---:|---:|---:|---:|
| `xgb_advanced_moe_postprocess` | XGB advanced + class-conditional regressor + position postprocess | 0.9615 | 0.9653 | 0.1217 | 0.4529 |
| `baseline_xgb_advanced` | XGB advanced baseline | 0.9615 | 0.9653 | 0.4208 | 0.7962 |
| `tuned_xgb_advanced_balanced` | XGB advanced tuned | 0.9231 | 0.9339 | 0.2497 | 0.5085 |
| `tuned_xgb_balanced` | XGB tuned | 0.8846 | 0.8960 | 0.3816 | 0.6856 |
| `baseline_xgb` | XGB baseline | 0.8462 | 0.8538 | 0.4198 | 0.7678 |
| `tuned_rf_balanced_refit` | RF tuned + refit | 0.8462 | 0.6308 | 0.3703 | 0.6157 |
| `baseline_rf` | RF baseline | 0.7308 | 0.6124 | 0.6846 | 0.9749 |
| `tuned_cnn1d_balanced` | CNN1D tuned | 0.6923 | 0.3439 | 1.1632 | 1.5307 |
| `tuned_mlp_balanced` | MLP tuned | 0.6154 | 0.2540 | 1.1690 | 1.5027 |

## Kết luận nhanh

- Tốt nhất nếu ưu tiên phân loại `num_damages`: `xgb_advanced_moe_postprocess` (đồng hạng với `baseline_xgb_advanced`)
- Tốt nhất nếu ưu tiên dự đoán vị trí: `xgb_advanced_moe_postprocess`
- Tốt nhất nếu cân bằng cả hai: `xgb_advanced_moe_postprocess`

Ghi chú:
- Ở đây "cân bằng cả hai" được hiểu theo trực giác là vừa giữ `F1 macro` cao, vừa giữ `pos_mae` thấp.
- Nếu dùng score đơn giản `0.5 * F1 + 0.5 * (1 / (1 + MAE))`, thì `xgb_advanced_moe_postprocess` đứng đầu rõ rệt.

## Xếp hạng theo F1 macro

- 1. `xgb_advanced_moe_postprocess`: `f1_macro=0.9653`, `pos_mae=0.1217`
- 2. `baseline_xgb_advanced`: `f1_macro=0.9653`, `pos_mae=0.4208`
- 3. `tuned_xgb_advanced_balanced`: `f1_macro=0.9339`, `pos_mae=0.2497`
- 4. `tuned_xgb_balanced`: `f1_macro=0.8960`, `pos_mae=0.3816`
- 5. `baseline_xgb`: `f1_macro=0.8538`, `pos_mae=0.4198`
- 6. `tuned_rf_balanced_refit`: `f1_macro=0.6308`, `pos_mae=0.3703`
- 7. `baseline_rf`: `f1_macro=0.6124`, `pos_mae=0.6846`
- 8. `tuned_cnn1d_balanced`: `f1_macro=0.3439`, `pos_mae=1.1632`
- 9. `tuned_mlp_balanced`: `f1_macro=0.2540`, `pos_mae=1.1690`

## Xếp hạng theo Pos MAE

- 1. `xgb_advanced_moe_postprocess`: `pos_mae=0.1217`, `f1_macro=0.9653`
- 2. `tuned_xgb_advanced_balanced`: `pos_mae=0.2497`, `f1_macro=0.9339`
- 3. `tuned_rf_balanced_refit`: `pos_mae=0.3703`, `f1_macro=0.6308`
- 4. `tuned_xgb_balanced`: `pos_mae=0.3816`, `f1_macro=0.8960`
- 5. `baseline_xgb`: `pos_mae=0.4198`, `f1_macro=0.8538`
- 6. `baseline_xgb_advanced`: `pos_mae=0.4208`, `f1_macro=0.9653`
- 7. `baseline_rf`: `pos_mae=0.6846`, `f1_macro=0.6124`
- 8. `tuned_cnn1d_balanced`: `pos_mae=1.1632`, `f1_macro=0.3439`
- 9. `tuned_mlp_balanced`: `pos_mae=1.1690`, `f1_macro=0.2540`

## Khuyến nghị thực tế

- Nếu bài toán của bạn ưu tiên nhận diện đúng số lượng hư hỏng trước: chọn `xgb_advanced_moe_postprocess` (đồng hạng F1 cao nhất với baseline advanced)
- Nếu bạn cần model cân bằng hơn giữa phân loại và vị trí: chọn `xgb_advanced_moe_postprocess`
- Nếu muốn một baseline cây đơn giản hơn XGBoost nhưng vị trí vẫn ổn: `tuned_rf_balanced_refit` là phương án phụ
- `MLP` và `CNN1D` hiện chưa phải ứng viên mạnh trong repo này

