# TÓM TẮT TIẾN ĐỘ BÁO CÁO - 29/04/2026

## 1. Mục tiêu hiện tại của đề tài
- Xây dựng pipeline để xác định hư hỏng dầm từ dữ liệu modal.
- Bài toán gồm 2 phần chính:
  - phân loại số lượng vị trí hư hỏng (`num_damages`)
  - dự đoán các vị trí hư hỏng (`damage_pos_1..4`)

## 2. Đã làm được tới đâu
- Đã xử lý xong pipeline dữ liệu:
  - build dataset
  - chia `train/val/test` theo kiểu tránh leakage
- Đã xây dựng và chạy được nhiều nhóm model:
  - RandomForest
  - XGBoost
  - MLP
  - CNN1D
- Đã triển khai tuning cho:
  - RandomForest
  - XGBoost
- Đã bổ sung feature nâng cao:
  - wavelet-inspired features
  - physics-inspired features
- Đã có hệ thống evaluate và error analysis trên `test`

## 3. Giải thích nhanh các metric trong `metrics_summary.json`
- `accuracy`:
  - Tỷ lệ dự đoán đúng nhãn `num_damages` trên toàn bộ mẫu.
  - Ví dụ `0.9615` nghĩa là đúng khoảng 96.15% số mẫu.
- `f1_macro`:
  - Trung bình F1 của từng lớp `num_damages` (mỗi lớp có trọng số ngang nhau).
  - Dùng metric này để tránh việc lớp nhiều mẫu lấn át lớp ít mẫu.
- `pos_mae_overall`:
  - Sai số tuyệt đối trung bình của dự đoán vị trí hư hỏng (`damage_pos_1..4`), có mask slot không hợp lệ.
  - Càng nhỏ càng tốt.
- `pos_rmse_overall`:
  - Căn bậc hai của sai số bình phương trung bình cho vị trí hư hỏng.
  - Nhạy hơn MAE với các mẫu sai lớn (outlier).
- Lỗi theo từng slot vị trí (`pos_mae_per_slot`, `pos_rmse_per_slot`):
  - Cho biết model dự đoán tốt/xấu ở slot nào (`damage_pos_1`, `damage_pos_2`, ...).
  - Hữu ích để phân tích vì bài toán có nhiều số lượng hư hỏng khác nhau.

## 4. Kết quả nổi bật nhất hiện tại (cập nhật hôm nay)
- Baseline ban đầu `baseline_rf`:
  - `acc=0.7308`
  - `f1_macro=0.6124`
  - `pos_mae=0.6846`
- Model mạnh trước đây (advanced baseline):
  - `baseline_xgb_advanced`
  - `acc=0.9615`
  - `f1_macro=0.9653`
  - `pos_mae=0.4208`
- Model tốt nhất hiện tại sau cải tiến thuật toán:
  - `xgb_advanced_moe_postprocess`
  - `acc=0.9615`
  - `f1_macro=0.9653`
  - `pos_mae=0.1217`
  - `pos_rmse=0.4529`
- Mức cải thiện chính:
  - So với `baseline_xgb_advanced`, MAE vị trí giảm từ `0.4208` xuống `0.1217` (giảm mạnh), trong khi F1 giữ nguyên.

## 5. Nhận xét ngắn
- Kết quả hiện tại đã vượt mức kỳ vọng ban đầu của giai đoạn baseline.
- XGBoost đang vượt rõ RandomForest.
- MLP và CNN1D hiện chưa phải hướng mạnh trong repo này.
- Run `xgb_advanced_moe_postprocess` hiện là ứng viên mạnh nhất để đưa vào báo cáo chính.

## 6. Nên xem file nào để kiểm tra
- `docs/KEEP_TRACK.md`: theo dõi tiến độ tổng thể
- `docs/MODEL_COMPARISON.md`: so sánh các model
- `outputs/xgb_advanced_moe_postprocess/metrics_summary.json`: run tốt nhất hiện tại (F1 cao + MAE vị trí thấp)
- `outputs/xgb_advanced_moe_postprocess/test_position_error_by_class.csv`: lỗi vị trí theo từng lớp số hư hỏng
- `outputs/xgb_advanced_moe_postprocess/test_top15_position_errors.csv`: các case sai vị trí nhiều nhất để phân tích

## 7. Hướng tiếp theo
- Chạy ablation ngắn cho model mới để tăng tính thuyết phục khoa học:
  - bỏ class-conditional regressor
  - bỏ bước postprocess/snap
  - so sánh lại MAE/RMSE/F1
- Chạy thêm nhiều random seed để kiểm tra độ ổn định thống kê của kết quả.
- Chốt model cuối cùng và dựng phần thảo luận kết quả theo 2 trục:
  - phân loại số lượng hư hỏng (`num_damages`)
  - định vị vị trí hư hỏng (`damage_pos_*`)
