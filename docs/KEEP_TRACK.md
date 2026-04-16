# KEEP TRACK - Tiến độ dự án

## Mục tiêu
Xây pipeline để dự đoán vị trí hư hỏng (beam damage localization) từ dữ liệu modal (tần số + mode shape), sau đó huấn luyện và đánh giá các baseline ML theo cách chia `train/val/test` không gây rò rỉ dữ liệu (leakage-safe).

## Cập nhật hôm nay
- Hôm nay là **Thứ Năm, ngày 16/04/2026**.
- Tính tới hôm nay, project **không còn dừng ở mức baseline RF** nữa.
- Phần đã làm tới:
  - hoàn thiện pipeline dữ liệu
  - train/evaluate nhiều nhóm model
  - tuning cho RF và XGBoost
  - thêm nhóm feature nâng cao
  - có tài liệu so sánh model trên `test`
- Trạng thái hiện tại:
  - pipeline nghiên cứu đã chạy được end-to-end
  - đã có đủ cơ sở để báo cáo tiến độ và chọn hướng model ưu tiên
  - model mạnh nhất hiện tại thuộc nhóm **XGBoost**, không còn là RF

## Tóm tắt rất ngắn
- Nếu chỉ nhìn kết quả hiện tại trên `test`, nhóm **XGBoost** đang tốt nhất.
- Nếu ưu tiên phân loại đúng số lượng hư hỏng (`num_damages`), run tốt nhất là `baseline_xgb_advanced`.
- Nếu ưu tiên dự đoán vị trí hư hỏng hoặc muốn cân bằng giữa phân loại và vị trí, run tốt nhất là `tuned_xgb_advanced_balanced`.
- RF vẫn là baseline tốt và dễ giải thích, nhưng hiện đã bị XGBoost vượt qua khá rõ.

## Pipeline hiện đang ở phase nào?

### Phase hiện tại: Advanced modeling / feature engineering
- Các phase nền tảng đã xong:
  - build dataset
  - split `train/val/test`
  - baseline feature matrix
  - baseline model đầu tiên
  - evaluate đúng trên `train/val/test`
- Hiện tại pipeline đã đi qua phase baseline-first và đang ở phase:
  - so sánh nhiều model (`RF`, `XGBoost`, `MLP`)
  - tuning model
  - thêm feature nâng cao (`wavelet`, `physics`)
- Run tốt nhất hiện tại trong repo:
  - nếu ưu tiên phân loại `num_damages`: `baseline_xgb_advanced`
  - nếu ưu tiên vị trí / cân bằng hơn: `tuned_xgb_advanced_balanced`
- File so sánh tổng hợp:
  - `docs/MODEL_COMPARISON.md`

## Những phần của pipeline đã có
- `inspect_dataset.py`: có
- build dataset: có
- split leakage-safe: có
- baseline features: có
- wavelet features: có
- physics-inspired features: có
- RandomForest baseline: có
- XGBoost baseline: có
- MLP baseline: có (mức smoke / thử nghiệm)
- CNN 1D baseline: có (mức smoke / thử nghiệm)
- tuning RF: có
- tuning XGBoost: có (mức smoke)
- evaluate theo artifact: có
- error analysis trên `test`: có

## Những phần còn thiếu hoặc chưa hoàn chỉnh
- **CNN 1D chưa phải model mạnh**:
  - đã có baseline và đã có tuning
  - nhưng kết quả hiện còn yếu hơn nhóm XGBoost
- **XGBoost đã được tune, nhưng còn có thể tune tiếp sâu hơn**:
  - `baseline_xgb` đã có `tuned_xgb_balanced`
  - `baseline_xgb_advanced` đã có `tuned_xgb_advanced_balanced`
  - nếu cần, vẫn có thể thử thêm alpha khác hoặc mở rộng search space
- **MLP chưa phải model mạnh**:
  - hiện mới ở mức pipeline chạy được
  - kết quả còn yếu, chưa phải ứng viên tốt
- **Tuning cho feature advanced**:
  - đã có script tuning và đã có run tuned
  - việc còn lại chủ yếu là chọn model cuối cùng theo mục tiêu bài toán
- **So sánh tổng hợp cuối cùng**:
  - hiện đã có `docs/MODEL_COMPARISON.md` để chốt tương đối rõ model tốt nhất theo từng mục tiêu
  - việc còn lại là chốt 1 model cuối cùng để dùng làm kết quả báo cáo/chương tiếp theo

## Đã làm xong

### 1) Tạo dataset + chia train/val/test (an toàn leakage)
- Tạo dataset dạng “scenario-level” (mỗi cấu hình hư hỏng là 1 mẫu).
- Chia `train/val/test` theo *configuration level* để tránh leakage.
- File đầu ra chính:
  - `data/processed/scenario_dataset.csv`
  - `data/processed/train.csv`
  - `data/processed/val.csv`
  - `data/processed/test.csv`
- Script liên quan:
  - `scripts/data/01_build_dataset.py`
  - `scripts/data/02_split_dataset.py`

### 2) Baseline RandomForest (end-to-end) + đánh giá test
- Huấn luyện baseline RandomForest trên ma trận feature baseline.
- Đánh giá trên `test`:
  - Phân loại `num_damages`
  - Đánh giá lỗi vị trí bằng MAE/RMSE (đã mask các slot không hợp lệ)
- Output (error analysis) chính:
  - `outputs/baseline_rf/artifact.joblib`
  - `outputs/baseline_rf/test_confusion_matrix.csv`
  - `outputs/baseline_rf/test_classification_report.txt`
  - `outputs/baseline_rf/test_position_error_by_class.csv`
  - `outputs/baseline_rf/test_top15_position_errors.csv`
- Script liên quan:
  - `scripts/train/03_train_baseline.py`
  - `scripts/analysis/04_error_analysis.py`
- Snapshot metric (baseline hiện tại, trên `test`):
  - `baseline_rf`: `acc=0.7308`, `f1_macro=0.6124`, `pos_mae_overall(masked)=0.6846`

### 3) Tuning RF (chỉ dùng `train/val`, không đụng `test`)
- Thêm script tuning RandomForest chạy nhiều cấu hình.
- Chọn “best” theo trade-off giữa:
  - `num_damages` macro-F1
  - vị trí `pos` MAE (mask)
- Script:
  - `scripts/tune/06_tune_rf.py`
- Output:
  - `outputs/tuned_rf/` (run cũ)
  - `outputs/tuned_rf_balanced/` (run theo trade-off cân bằng)
- Snapshot metric (best chọn theo `val`, trên `test`):
  - `tuned_rf`: `acc=0.6923`, `f1_macro=0.5360`, `pos_mae_overall(masked)=0.4732`
  - `tuned_rf_balanced`: trong run này trùng best với `tuned_rf` (chọn hyperparameter tương tự)

### 4) Train lại best RF trên `train+val` rồi đánh giá `test` (không leakage)
- Lấy hyperparameter tốt nhất từ `tuned_rf_balanced`.
- Train lại (refit) trên `train + val`.
- Sau đó mới đánh giá trên `test`.
- Output:
  - `outputs/tuned_rf_balanced_refit/artifact.joblib`
  - `outputs/tuned_rf_balanced_refit/test_*.csv/txt`
- Snapshot metric (trên `test` sau refit):
  - `tuned_rf_balanced_refit`: `acc=0.8462`, `f1_macro=0.6308`, `pos_mae_overall(masked)=0.3703`

### 5) Baseline XGBoost (train + error analysis trên test)
- Triển khai script train baseline XGBoost và dùng `scripts/analysis/04_error_analysis.py` để xuất báo cáo trên `test`.
- Lưu artifact:
  - `outputs/baseline_xgb/artifact.joblib`
  - báo cáo/CSV trong `outputs/baseline_xgb/`
- Lưu ý kỹ thuật: trong `src/models/baseline_xgb.py` đã thêm cơ chế **remap nhãn `num_damages` về dạng liên tục** khi train (để tránh lỗi khác tập lớp giữa `train` và `val`).
- Snapshot metric trên `test` (baseline_xgb):
  - `acc(num_damages)=0.8462`
  - `f1_macro(num_damages)=0.8538`
  - `pos_mae_overall(masked)=0.4198`

### 6) Baseline MLP (smoke2)
- Đã triển khai baseline MLP (dùng `StandardScaler` cho ổn định huấn luyện).
- Chạy smoke với cấu hình tham số nhỏ để kiểm pipeline chạy đúng end-to-end.
- Artifact:
  - `outputs/baseline_mlp_smoke2/artifact.joblib`
- Snapshot metric trên `test`:
  - `acc(num_damages)=0.6923`
  - `f1_macro(num_damages)=0.2727`
  - `pos_mae_overall(masked)=1.2924`

### 7) Tuning XGBoost (smoke: train/val only, alpha=0.5)
- Thêm script random search tuning XGBoost chọn best dựa trên trade-off:
  - macro-F1 (`num_damages`)
  - vị trí MAE (masked, `damage_pos_*`)
- Script: `scripts/tune/07_tune_xgb.py` (smoke chạy `--n-trials 10 --alpha 0.5`)
- Artifact:
  - `outputs/tuned_xgb_balanced_smoke/artifact.joblib`
- Snapshot metric trên `test`:
  - `acc(num_damages)=0.8846`
  - `f1_macro(num_damages)=0.8960`
  - `pos_mae_overall(masked)=0.4422`

### 8) Feature nâng cao: wavelet + physics + XGBoost
- Đã triển khai:
  - `src/features/wavelet_features.py`
  - `src/features/physics_features.py`
  - `scripts/train/08_train_xgb_advanced.py`
- Ý tưởng:
  - giữ baseline features hiện có
  - cộng thêm wavelet-inspired features
  - cộng thêm physics-inspired features (gradient, curvature, zero-crossing, peak count...)
- Run hiện tại:
  - `outputs/baseline_xgb_advanced/`
- Snapshot metric trên `test`:
  - `acc(num_damages)=0.9615`
  - `f1_macro(num_damages)=0.9653`
  - `pos_mae_overall(masked)=0.4208`
- Nhận xét nhanh:
  - phần phân loại `num_damages` cải thiện rõ so với `baseline_xgb`
  - lỗi vị trí tổng thể hiện gần tương đương `baseline_xgb`, chưa giảm mạnh

### 9) Baseline CNN 1D (smoke)
- Đã triển khai:
  - `src/models/cnn1d.py`
  - `scripts/train/09_train_cnn1d.py`
- Ý tưởng:
  - dùng trực tiếp 4 mode vectors làm 4 kênh đầu vào
  - ghép thêm 4 modal frequencies
  - 2 đầu ra:
    - phân loại `num_damages`
    - hồi quy `damage_pos_1..4`
- Run hiện tại:
  - `outputs/baseline_cnn1d_smoke/`
- Snapshot metric trên `test`:
  - `acc(num_damages)=0.6923`
  - `f1_macro(num_damages)=0.2727`
  - `pos_mae_overall(masked)=1.0958`
- Nhận xét nhanh:
  - pipeline CNN đã chạy được end-to-end
  - nhưng chất lượng hiện còn thấp hơn rõ rệt so với XGBoost

### 10) Tuning full cho XGBoost advanced
- Đã triển khai:
  - `scripts/tune/10_tune_xgb_advanced.py`
- Thiết lập run:
  - tuning trên `train/val`
  - tiêu chí chọn best theo trade-off `macro-F1 + position MAE`
  - run hiện tại dùng `--n-trials 30 --alpha 0.5 --refit-on-train-val`
- Artifact:
  - `outputs/tuned_xgb_advanced_balanced/artifact.joblib`
- Best hyperparameters hiện tại:
  - `n_estimators=800`
  - `learning_rate=0.03`
  - `max_depth=6`
  - `subsample=1.0`
  - `colsample_bytree=1.0`
  - `reg_lambda=3.0`
- Snapshot metric trên `test`:
  - `acc(num_damages)=0.9231`
  - `f1_macro(num_damages)=0.9339`
  - `pos_mae_overall(masked)=0.2497`
- Nhận xét nhanh:
  - model này **giảm lỗi vị trí rất mạnh** so với `baseline_xgb_advanced`
  - nhưng phần phân loại `num_damages` thấp hơn `baseline_xgb_advanced`
  - vì run này có `--refit-on-train-val`, metric `VAL` sau refit không còn dùng như validation độc lập; nên khi so sánh, ưu tiên nhìn `TEST`

## So sánh model hiện tại

### Model đang tốt nhất theo từng mục tiêu
- Nếu ưu tiên **phân loại đúng số lượng hư hỏng**:
  - chọn `baseline_xgb_advanced`
  - kết quả `test`: `acc=0.9615`, `f1_macro=0.9653`, `pos_mae=0.4208`
- Nếu ưu tiên **dự đoán vị trí hư hỏng chính xác hơn**:
  - chọn `tuned_xgb_advanced_balanced`
  - kết quả `test`: `acc=0.9231`, `f1_macro=0.9339`, `pos_mae=0.2497`
- Nếu cần **cân bằng cả hai**:
  - hiện tại vẫn nên chọn `tuned_xgb_advanced_balanced`
  - lý do: F1 vẫn rất cao nhưng MAE vị trí giảm mạnh nhất

### So với các nhóm model khác
- `baseline_rf`:
  - `acc=0.7308`, `f1_macro=0.6124`, `pos_mae=0.6846`
- `tuned_rf_balanced_refit`:
  - `acc=0.8462`, `f1_macro=0.6308`, `pos_mae=0.3703`
- `baseline_xgb`:
  - `acc=0.8462`, `f1_macro=0.8538`, `pos_mae=0.4198`
- `tuned_xgb_balanced`:
  - `acc=0.8846`, `f1_macro=0.8960`, `pos_mae=0.3816`
- `baseline_xgb_advanced`:
  - `acc=0.9615`, `f1_macro=0.9653`, `pos_mae=0.4208`
- `tuned_xgb_advanced_balanced`:
  - `acc=0.9231`, `f1_macro=0.9339`, `pos_mae=0.2497`
- `MLP` và `CNN1D` hiện thấp hơn rõ rệt, chưa phải ứng viên chính

## Kết quả hiện tại so với kỳ vọng ban đầu

### Kỳ vọng ban đầu
- Mục tiêu ban đầu là dựng được một pipeline sạch:
  - đọc dữ liệu
  - build dataset
  - split không leakage
  - train được baseline đầu tiên
  - evaluate đúng trên `test`
- Kỳ vọng tiếp theo là thử thêm nhiều model để xem có vượt được baseline RF hay không.

### Kết quả thực tế đến hôm nay
- Đã **vượt kỳ vọng ban đầu**.
- Không chỉ dừng ở mức pipeline baseline:
  - đã có nhiều model chạy được end-to-end
  - đã có tuning
  - đã có feature engineering nâng cao
  - đã có bảng so sánh model
- Kết quả nổi bật nhất:
  - từ `baseline_rf` lên `baseline_xgb_advanced`, `f1_macro` tăng từ `0.6124` lên `0.9653`
  - từ `baseline_rf` lên `tuned_xgb_advanced_balanced`, `pos_mae` giảm từ `0.6846` xuống `0.2497`
- Nghĩa là:
  - phần phân loại số lượng hư hỏng đã tốt lên rất nhiều
  - phần dự đoán vị trí hư hỏng cũng đã cải thiện rõ

### Điều vẫn chưa đạt như mong muốn
- Chưa chốt được một model cuối cùng theo đúng ưu tiên bài toán:
  - nếu ưu tiên F1 cao nhất thì chọn một model
  - nếu ưu tiên vị trí tốt nhất thì chọn model khác
- `MLP` và `CNN1D` chưa cho kết quả mạnh như kỳ vọng.
- Vẫn còn dư địa tuning sâu hơn cho XGBoost advanced nếu muốn tối ưu thêm.

## Muốn xem kết quả thì mở file nào?

### File tổng hợp nên xem đầu tiên
- `docs/MODEL_COMPARISON.md`
  - file này cho biết model nào đang tốt nhất theo từng tiêu chí
- `docs/KEEP_TRACK.md`
  - file này cho biết tiến độ đã làm tới đâu và logic phát triển của project

### File để kiểm tra metric từng run
- `outputs/baseline_rf/metrics_summary.json`
- `outputs/tuned_rf_balanced_refit/metrics_summary.json`
- `outputs/baseline_xgb/metrics_summary.json`
- `outputs/tuned_xgb_balanced/metrics_summary.json`
- `outputs/baseline_xgb_advanced/metrics_summary.json`
- `outputs/tuned_xgb_advanced_balanced/metrics_summary.json`

### File để xem chi tiết lỗi
- `outputs/<run_name>/test_classification_report.txt`
  - xem precision / recall / f1 theo từng lớp
- `outputs/<run_name>/test_confusion_matrix.csv`
  - xem model nhầm lớp nào với lớp nào
- `outputs/<run_name>/test_position_error_by_class.csv`
  - xem lỗi vị trí theo từng số lượng hư hỏng
- `outputs/<run_name>/test_top15_position_errors.csv`
  - xem các mẫu sai vị trí nhiều nhất

## Tóm tắt để báo cáo thầy

### Bản ngắn gọn
- Đề tài đã xây dựng xong pipeline xử lý dữ liệu và huấn luyện mô hình để xác định hư hỏng dầm từ dữ liệu modal.
- Pipeline hiện đã chạy được đầy đủ từ bước tạo dataset, chia `train/val/test`, trích xuất feature, huấn luyện model, đến đánh giá trên `test`.
- Ban đầu dùng RandomForest làm baseline, sau đó mở rộng sang XGBoost, MLP và CNN1D.
- Kết quả tốt nhất hiện tại thuộc về nhóm XGBoost.
- Nếu ưu tiên phân loại đúng số lượng hư hỏng thì `baseline_xgb_advanced` đang tốt nhất với `f1_macro=0.9653`.
- Nếu ưu tiên dự đoán vị trí hư hỏng thì `tuned_xgb_advanced_balanced` đang tốt nhất với `pos_mae=0.2497`.
- Nhìn chung, kết quả hiện tại đã vượt mức kỳ vọng ban đầu của giai đoạn baseline và đã đi sang giai đoạn tối ưu model.

### Bản nói dễ hiểu hơn
- Lúc đầu mục tiêu chỉ là làm được một pipeline baseline chạy đúng và đánh giá đúng.
- Đến nay, ngoài baseline RF, project đã có thêm nhiều mô hình khác và đã chứng minh được rằng XGBoost cho kết quả tốt hơn khá rõ.
- Hiện tại có thể xem như bài toán đã có 2 hướng chốt:
  - một model mạnh về phân loại số lượng hư hỏng
  - một model cân bằng hơn giữa phân loại và định vị vị trí hư hỏng
- Bước tiếp theo hợp lý là chọn 1 tiêu chí ưu tiên chính, rồi chốt model cuối cùng để viết báo cáo hoặc phát triển tiếp.

## Các file đã thêm/cập nhật
- Thêm: `scripts/tune/06_tune_rf.py`
- Cập nhật: `scripts/analysis/04_error_analysis.py` (thêm tham số `--run-name` để dùng cho nhiều run)
- Cập nhật: `src/models/baseline_sklearn.py` (mở rộng cấu hình hyperparameter cho RF)
- Thêm: `scripts/train/03_train_xgb.py` (XGBoost baseline)
- Sửa: `src/models/baseline_xgb.py` (remap nhãn lớp)
- Thêm: `src/models/baseline_mlp.py` và `scripts/train/04_train_mlp.py` (MLP baseline)
- Thêm: `scripts/tune/07_tune_xgb.py` (tuning XGBoost)
- Thêm: `scripts/analysis/05_evaluate.py` (tính metrics nhanh theo `artifact.joblib`)
- Thêm: `src/features/wavelet_features.py`
- Thêm: `src/features/physics_features.py`
- Thêm: `scripts/train/08_train_xgb_advanced.py`
- Thêm: `src/models/cnn1d.py`
- Thêm: `scripts/train/09_train_cnn1d.py`
- Thêm: `scripts/tune/10_tune_xgb_advanced.py`

## Cách chạy lại (gợi ý)
- Train baseline RF:
  - `python scripts/train/03_train_baseline.py`
  - `python scripts/analysis/04_error_analysis.py --run-name baseline_rf`
- Tuning RF (chọn cân bằng):
  - `python scripts/tune/06_tune_rf.py --n-trials 80 --output-name tuned_rf_balanced --alpha 0.5`
  - `python scripts/analysis/04_error_analysis.py --run-name tuned_rf_balanced`
- Nếu muốn refit luôn (không đụng test khi chọn best):
  - `python scripts/tune/06_tune_rf.py --n-trials 80 --output-name tuned_rf_balanced --alpha 0.5 --refit-on-train-val`

## Việc tiếp theo (chưa làm)
- Tuning cho `CNN 1D` nếu muốn tiếp tục theo hướng deep learning.
- Nếu muốn tiếp: thử alpha khác cho tuning `XGBoost advanced` để xem có giữ được F1 cao hơn mà MAE vẫn thấp.
- Nếu muốn tiếp: dọn lại các run smoke/old run trong `outputs/` và chỉ giữ lại các run cuối cùng quan trọng.

