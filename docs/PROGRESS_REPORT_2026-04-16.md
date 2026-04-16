# TÓM TẮT TIẾN ĐỘ BÁO CÁO - 16/04/2026

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

## 3. Kết quả nổi bật nhất hiện tại
- Baseline ban đầu `baseline_rf`:
  - `acc=0.7308`
  - `f1_macro=0.6124`
  - `pos_mae=0.6846`
- Model mạnh nhất nếu ưu tiên phân loại:
  - `baseline_xgb_advanced`
  - `acc=0.9615`
  - `f1_macro=0.9653`
  - `pos_mae=0.4208`
- Model tốt nhất nếu ưu tiên vị trí hoặc cân bằng:
  - `tuned_xgb_advanced_balanced`
  - `acc=0.9231`
  - `f1_macro=0.9339`
  - `pos_mae=0.2497`

## 4. Nhận xét ngắn
- Kết quả hiện tại đã vượt mức kỳ vọng ban đầu của giai đoạn baseline.
- XGBoost đang vượt rõ RandomForest.
- MLP và CNN1D hiện chưa phải hướng mạnh trong repo này.
- Hiện tại có thể coi project đã qua giai đoạn dựng baseline và đang ở giai đoạn chọn model cuối cùng.

## 5. Nên xem file nào để kiểm tra
- `docs/KEEP_TRACK.md`: theo dõi tiến độ tổng thể
- `docs/MODEL_COMPARISON.md`: so sánh các model
- `outputs/baseline_xgb_advanced/metrics_summary.json`: kết quả model mạnh về F1
- `outputs/tuned_xgb_advanced_balanced/metrics_summary.json`: kết quả model mạnh về vị trí/cân bằng

## 6. Hướng tiếp theo
- Chốt tiêu chí ưu tiên chính:
  - ưu tiên phân loại đúng số lượng hư hỏng
  - hoặc ưu tiên định vị vị trí hư hỏng tốt hơn
- Từ đó chọn model cuối cùng để dùng cho báo cáo và phần thảo luận kết quả.
