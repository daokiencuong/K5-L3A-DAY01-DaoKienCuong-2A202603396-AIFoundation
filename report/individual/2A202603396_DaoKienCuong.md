# BÁO CÁO CÁ NHÂN — DAY 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603396
- **Họ và tên:** Đào Kiên Cường
- **SEED:** 314

## 1. Bằng chứng

| Tệp | Có? |
|---|---|
| `day01_spam_ml_dl_embedding_llm.ipynb` — Notebook đã chạy, còn output | Có |
| `lab1_report.json` — Tệp được sinh từ ô “Báo cáo cuối” | Có |
| `hand_calc.jpg` — Ảnh tính tay có các bước | Có |

## 2. Kết quả chính

Bảng dưới đây tổng hợp kết quả đánh giá các mô hình trên tập test và tập SUB, đồng thời thể hiện thời gian huấn luyện, độ trễ suy luận, tốc độ xử lý và chi phí ước tính.

| Mô hình | Accuracy | F1 test | F1 SUB | Train (s) | p50 (ms) | p95 (ms) | Tin/s | USD/1k tin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BoW + LogReg | 0.979 | 0.908 | 0.947 | 4.58 | 0.051 | 0.086 | 42682.55 | 0 |
| BoW + MLP | 0.983 | 0.929 | 0.950 | 12.59 | 0.278 | 0.720 | 7868.76 | 0 |
| MiniLM + kNN | 0.981 | 0.924 | 0.974 | 194.50 | 124.57 | 263.08 | 11.17 | 0 |
| MiniLM + LogReg | 0.992 | 0.968 | 0.947 | 196.94 | 55.40 | 150.30 | 13.67 | 0 |
| MiniLM + MLP head | 0.992 | 0.968 | 0.947 | 198.17 | 55.12 | 175.19 | 11.09 | 0 |
| LLM nhỏ zero-shot | 0.500 | 0.000 | 0.000 | 0 | 715.83 | 1631.50 | 0.96 | 0.00488 |
| LLM lớn zero-shot | 0.500 | 0.000 | 0.000 | 0 | 882.27 | 1295.85 | 1.03 | 0.04877 |
| LLM nhỏ few-shot | 0.500 | 0.000 | 0.000 | 0 | 678.71 | 1340.85 | 1.26 | 0.01221 |

**Nhận xét:**

- MiniLM + LogReg và MiniLM + MLP head đạt F1 test cao nhất, cùng ở mức 0.968. Điều này cho thấy biểu diễn ngữ nghĩa từ MiniLM có khả năng hỗ trợ phân loại tốt trên tập test.
- Trong nhóm BoW, MLP đạt F1 test 0.929, cao hơn Logistic Regression (0.908), nhưng thời gian huấn luyện và độ trễ suy luận cũng lớn hơn.
- MiniLM + kNN đạt F1 SUB cao nhất, ở mức 0.974. Tuy nhiên, p95 lên tới 263.08 ms, cao hơn đáng kể so với yêu cầu SLA 50 ms.
- Các cấu hình LLM trong lần thử nghiệm này đều có F1 test và F1 SUB bằng 0.000. Đây là kết quả bất thường cần kiểm tra thêm, đặc biệt ở bước xử lý và chuyển đổi đầu ra thành nhãn.

Nhìn chung, các mô hình có sự khác biệt rõ rệt về chất lượng phân loại và hiệu năng. Mô hình có F1 cao nhất chưa chắc là lựa chọn phù hợp nhất khi triển khai thực tế, bởi còn phải xem xét độ trễ, tốc độ xử lý và chi phí.

## 3. Dự đoán và thực tế

Trước khi thực hiện các thử nghiệm, em đã đưa ra nhiều dự đoán về chất lượng phân loại, tốc độ xử lý và chi phí của các mô hình. Sau khi chạy thực nghiệm, em tiến hành đối chiếu các nhận định ban đầu với kết quả thu được.

### 1. Dự đoán về LLM lớn zero-shot

- **Em dự đoán:** LLM lớn có khả năng hiểu ngữ nghĩa tốt hơn và có thể đạt F1 trong khoảng 0.93–0.97.
- **Kết quả thực tế:** F1 SUB của LLM lớn zero-shot bằng 0.000, thấp hơn rất nhiều so với dự đoán ban đầu.
- **Vì sao lệch:** Em đã kỳ vọng quá nhiều vào khả năng hiểu ngữ nghĩa của LLM mà chưa xem xét đầy đủ quy trình trả về và xử lý nhãn. Một khả năng cần kiểm tra là phản hồi của LLM không đúng định dạng chương trình yêu cầu, dẫn đến lỗi khi phân tích hoặc chuyển đổi nhãn. Vì vậy, cần kiểm tra đầu ra gốc trước khi kết luận về khả năng phân loại của mô hình.

### 2. Dự đoán về phương pháp Few-shot

- **Em dự đoán:** Việc bổ sung một số ví dụ vào prompt sẽ giúp LLM cải thiện F1 nhẹ, nhưng chi phí có thể tăng khoảng 3–4 lần so với zero-shot.
- **Kết quả thực tế:** F1 SUB vẫn bằng 0.000. Chi phí tăng từ 0.00488 USD lên 0.01221 USD/1.000 tin, tương đương khoảng 2.5 lần.
- **Vì sao lệch:** Few-shot không tạo ra cải thiện về F1 trong lần thử nghiệm này, dù chi phí xử lý tăng lên. Điều đó cho thấy việc bổ sung ví dụ vào prompt không bảo đảm kết quả phân loại tốt hơn. Chất lượng ví dụ, cách thiết kế prompt và định dạng đầu ra đều có thể ảnh hưởng đến kết quả.

### 3. Dự đoán về đường Pareto

- **Em dự đoán:** BoW + Logistic Regression và Encoder + Logistic Regression có khả năng nằm trên đường Pareto do có sự cân bằng giữa chất lượng phân loại và hiệu năng.
- **Kết quả thực tế:** Các mô hình nằm trên đường Pareto được ghi nhận gồm BoW + Logistic Regression, BoW + MLP và MiniLM + kNN.
- **Vì sao lệch:** Em đã đánh giá chưa đầy đủ sự đánh đổi giữa F1, độ trễ và các tiêu chí hiệu năng. Một mô hình có F1 cao chưa chắc nằm trên đường Pareto nếu có mô hình khác đạt kết quả tốt hơn ở các tiêu chí liên quan mà không phải đánh đổi quá nhiều. Vì vậy, cần đánh giá đồng thời nhiều chỉ số thay vì chỉ dựa vào chất lượng phân loại.

### Các dự đoán khác

| Nội dung | Dự đoán ban đầu | Kết quả thực tế |
|---|---|---|
| F1 Logistic Regression | Khoảng 0.90 | F1 test đạt 0.908, khá sát dự đoán. |
| Learning rate | 0.001 | Giá trị được ghi nhận là 0.001. |
| Learning curve MLP | Validation loss giảm ban đầu rồi chững lại hoặc tăng; MLP có thể tốt hơn LR nhưng có nguy cơ overfit. | MLP đạt F1 test 0.929, cao hơn LR ở mức 0.908. Chưa đủ căn cứ kết luận overfit chỉ từ F1. |
| Bỏ ReLU | Hai lớp tuyến tính liên tiếp tương đương một lớp tuyến tính. | Nhận định đúng về mặt toán học. |
| Tin “complimentary holiday” | Dự đoán là spam do có ngữ nghĩa mồi nhử. | JSON không lưu nhãn dự đoán thực tế riêng cho tin nhắn này nên chưa thể xác nhận. |
| Tin nhắn ngụy trang | Encoder + LogReg sẽ phân loại đúng nhiều hơn BoW. | F1 SUB của Encoder + LogReg là 0.947, bằng BoW + LR và thấp hơn BoW + MLP (0.950). Dự đoán chưa được xác nhận. |
| 50 case khó | BoW đạt khoảng 30–50%; embedding/LLM đạt 60–80%. | BoW đạt Accuracy 49.1–52.7%; embedding đạt 54.5–63.6%; LLM nhỏ đạt 41.8%. |

**Kết luận:**

Qua quá trình đối chiếu, em nhận thấy một số dự đoán ban đầu tương đối sát với kết quả thực nghiệm, nhưng cũng có những nhận định sai lệch đáng kể, đặc biệt là kỳ vọng về LLM và việc xác định đường Pareto. Điều này cho thấy dự đoán chỉ có vai trò định hướng, còn kết luận cuối cùng phải dựa trên số liệu thực tế. Bên cạnh chất lượng phân loại, cần kiểm tra cả quy trình xử lý đầu ra, độ trễ và chi phí để đánh giá chính xác khả năng triển khai của mô hình.

## 4. Phân tích 50 case khó

Để đánh giá khả năng xử lý các tình huống dễ gây nhầm lẫn, em sử dụng tập gồm 50 case khó với nhiều nhóm nội dung khác nhau. Kết quả cho thấy các mô hình có sự khác biệt đáng kể về khả năng nhận diện.

| Mô hình | F1 50 case | Accuracy 50 case |
|---|---:|---:|
| BoW + LogReg | 0.333 | 0.491 |
| BoW + MLP | 0.409 | 0.527 |
| MiniLM + kNN | 0.677 | 0.636 |
| MiniLM + LogReg | 0.528 | 0.545 |
| MiniLM + MLP head | 0.556 | 0.564 |
| LLM nhỏ zero-shot | 0.000 | 0.418 |
| LLM nhỏ + prompt phòng thủ | 0.000 | 0.418 |

**Tỷ lệ dự đoán đúng theo từng nhóm case:**

| Nhóm case | BoW LogReg | BoW MLP | MiniLM kNN | MiniLM LogReg | MiniLM MLP |
|---|---:|---:|---:|---:|---:|
| Ham giống spam (10) | 100% | 100% | 100% | 100% | 100% |
| Spam tự nhiên (10) | 0% | 0% | 80% | 40% | 50% |
| Thao túng/lừa đảo (8) | 0% | 0% | 75% | 38% | 50% |
| Né từ khóa (6) | 0% | 17% | 50% | 33% | 33% |
| Prompt injection (6) | 0% | 17% | 33% | 17% | 33% |
| Nói về spam (4) | 100% | 100% | 100% | 100% | 100% |
| Ham giao dịch (6) | 100% | 100% | 100% | 100% | 100% |
| **Tổng 50 case (Accuracy)** | **49%** | **53%** | **64%** | **55%** | **56%** |

**Nhận xét về các nhóm case:**

- **Ham giống spam:** Tất cả các mô hình trong bảng đều đạt tỷ lệ đúng 100%. Điều này cho thấy các mô hình đã xử lý tốt nhóm tin nhắn hợp lệ có chứa từ ngữ dễ gây hiểu nhầm là spam trong bộ dữ liệu này.
- **Spam tự nhiên:** BoW + LogReg và BoW + MLP đều đạt 0%, trong khi MiniLM + kNN đạt 80%. Các tin nhắn spam có cách diễn đạt tự nhiên, ít phụ thuộc vào từ khóa quảng cáo quen thuộc, khiến mô hình dựa trên Bag-of-Words gặp khó khăn.
- **Thao túng/lừa đảo:** MiniLM + kNN đạt 75%, cao hơn các mô hình còn lại. Nhóm này thường sử dụng cách diễn đạt tạo áp lực hoặc thúc giục người nhận hành động, đòi hỏi mô hình phải nhận biết được ngữ cảnh và ý đồ của nội dung.
- **Né từ khóa:** Các mô hình BoW có kết quả thấp, đặc biệt Logistic Regression đạt 0%. Điều này phù hợp với hạn chế của phương pháp dựa trên tần suất và sự xuất hiện của từ khóa.
- **Prompt injection:** Đây vẫn là nhóm khó đối với tất cả các mô hình. MiniLM + kNN đạt 33%, cho thấy việc nhận diện nội dung có chỉ dẫn gây nhiễu hoặc cố tình đánh lừa hệ thống vẫn là thách thức.
- **Nói về spam và ham giao dịch:** Các mô hình trong bảng đều đạt 100%, cho thấy khả năng nhận diện tốt hai nhóm này trong tập thử nghiệm.

**Phân tích tổng thể:**

MiniLM + kNN là mô hình có kết quả tốt nhất trên 50 case khó, với F1 đạt 0.677 và Accuracy đạt 0.636. Kết quả này cho thấy phương pháp biểu diễn ngữ nghĩa có lợi thế trong việc nhận diện các tin nhắn spam không sử dụng những từ khóa quen thuộc.

Trong khi đó, hai mô hình BoW có kết quả thấp hơn, đặc biệt ở các nhóm spam tự nhiên, thao túng và né từ khóa. Nguyên nhân có thể đến từ việc Bag-of-Words chủ yếu dựa trên sự xuất hiện của token mà chưa thể nắm bắt đầy đủ ngữ cảnh của câu.

Đáng chú ý, LLM nhỏ zero-shot và LLM nhỏ sử dụng prompt phòng thủ đều có F1 bằng 0 và Accuracy bằng 0.418. Đây là kết quả cần được kiểm tra thêm ở bước xử lý phản hồi và phân tích nhãn. Không nên chỉ dựa trên các chỉ số này để kết luận rằng LLM hoàn toàn không có khả năng phân loại.

**Kết luận:**

Kết quả trên 50 case khó cho thấy F1 cao trên tập test thông thường không đồng nghĩa với khả năng xử lý tốt mọi tình huống. Mỗi nhóm dữ liệu có đặc điểm riêng và yêu cầu mô hình phải nhận diện được cả từ khóa lẫn ngữ cảnh. Do đó, khi xây dựng hệ thống lọc spam thực tế, cần bổ sung dữ liệu đa dạng và đánh giá riêng các nhóm trường hợp dễ gây nhầm lẫn.

## 5. Thử thách cá nhân

### 5.1. Xác định đề bài

Theo cách chọn thử thách của bài thực hành, em thực hiện **đề số 3**: thay đổi số nơ-ron ở lớp ẩn của MLP thành hai cấu hình 4 và 512. Yêu cầu gồm tính số lượng tham số, đo F1 test, thời gian huấn luyện và giải thích khi nào nên sử dụng mạng nhỏ.

### 5.2. Kết quả

Đoạn mã tham khảo:

```python
import time
import torch
import pandas as pd

challenge_rows = []

for hidden in [4, 512]:
    torch.manual_seed(SEED)
    model = build_mlp(X_tr.shape[1], hidden=hidden)

    start = time.perf_counter()
    train_mlp(
        model, X_tr, y_tr, X_va, y_va,
        epochs=15, lr=1e-3, batch_size=64
    )
    elapsed = time.perf_counter() - start

    pred = predict_mlp(model, X_te)
    score = metrics(y_te, pred)

    challenge_rows.append({
        "hidden": hidden,
        "parameters": sum(
            p.numel() for p in model.parameters()
        ),
        "F1_test": score["f1"],
        "train_seconds": elapsed
    })

display(pd.DataFrame(challenge_rows))
```

**Giải thích xu hướng:**

MLP với hidden = 4 chỉ có 8.009 tham số, do đó có ưu điểm là kiến trúc gọn nhẹ, ít tốn bộ nhớ và tài nguyên tính toán. Tuy nhiên, số lượng nơ-ron ít có thể làm giảm khả năng biểu diễn các quan hệ phức tạp trong dữ liệu, dẫn đến nguy cơ underfitting.

Ngược lại, MLP với hidden = 512 có 1.025.537 tham số, lớn hơn rất nhiều so với cấu hình hidden = 4. Mạng lớn có khả năng học được nhiều đặc trưng và quan hệ phức tạp hơn, nhưng cũng cần nhiều tài nguyên hơn và có nguy cơ overfitting nếu dữ liệu huấn luyện không đủ lớn hoặc không đa dạng.

Vì vậy, mạng nhỏ phù hợp khi dữ liệu tương đối đơn giản, tài nguyên triển khai hạn chế hoặc hệ thống yêu cầu tốc độ xử lý cao. Nếu mạng nhỏ vẫn đạt F1 validation gần tương đương mạng lớn thì đây có thể là lựa chọn hiệu quả hơn về chi phí vận hành.

Tuy nhiên, để đưa ra kết luận chính xác giữa hai cấu hình, cần chạy thực nghiệm và so sánh F1 test cùng thời gian huấn luyện. Số lượng tham số chỉ phản ánh quy mô mô hình, không trực tiếp chứng minh cấu hình nào có chất lượng phân loại tốt hơn.

## 6. Phương án cho sản phẩm thật

Với yêu cầu hệ thống xử lý 10 triệu SMS mỗi ngày và độ trễ p95 nhỏ hơn 50 ms/tin, em lựa chọn BoW + MLP làm phương án ưu tiên trong hai mô hình BoW đáp ứng SLA.

Cụ thể, BoW + MLP đạt F1 test 0.929, Accuracy 0.983 và p95 = 0.720 ms. Trong khi đó, BoW + Logistic Regression có F1 test 0.908, Accuracy 0.979 và p95 = 0.086 ms. Như vậy, MLP có chất lượng phân loại cao hơn, còn Logistic Regression có độ trễ thấp hơn và thời gian huấn luyện ngắn hơn.

**Đánh giá theo yêu cầu SLA:**

| Mô hình | F1 test | p95 (ms) | Đáp ứng p95 < 50 ms? |
|---|---:|---:|---|
| BoW + LogReg | 0.908 | 0.086 | Có |
| BoW + MLP | 0.929 | 0.720 | Có |
| MiniLM + kNN | 0.924 | 263.08 | Không |
| MiniLM + LogReg | 0.968 | 150.30 | Không |
| MiniLM + MLP head | 0.968 | 175.19 | Không |
| LLM nhỏ zero-shot | 0.000 | 1631.50 | Không |
| LLM lớn zero-shot | 0.000 | 1295.85 | Không |
| LLM nhỏ few-shot | 0.000 | 1340.85 | Không |

Với lưu lượng 10 triệu tin nhắn mỗi ngày, hệ thống cần xử lý trung bình:

\[
\frac{10.000.000}{24 \times 60 \times 60}
\approx 115,7 \text{ tin/giây}
\]

Trong hai mô hình đáp ứng SLA, BoW + MLP có F1 test cao hơn nên phù hợp khi ưu tiên chất lượng phân loại. Nếu cần tối ưu tốc độ, giảm độ phức tạp và thời gian huấn luyện, BoW + Logistic Regression là phương án thay thế hợp lý.

Các mô hình MiniLM có F1 test cao hơn, đạt 0.968, nhưng độ trễ p95 đều vượt 50 ms trong lần đo này. Vì vậy, chưa thể lựa chọn chúng làm mô hình xử lý trực tiếp toàn bộ lưu lượng nếu SLA là điều kiện bắt buộc.

**Ước tính chi phí sử dụng LLM:**

Với 10 triệu tin nhắn mỗi ngày, tương đương 10.000 đơn vị 1.000 tin nhắn, chi phí được ước tính dựa trên chỉ số `usd_per_1k` trong báo cáo:

| Cấu hình | USD/1.000 tin | USD/ngày | USD/30 ngày |
|---|---:|---:|---:|
| LLM nhỏ zero-shot | 0.0048767 | 48.77 | 1,463.00 |
| LLM lớn zero-shot | 0.0487667 | 487.67 | 14,630.00 |
| LLM nhỏ few-shot | 0.0122100 | 122.10 | 3,663.00 |

Kết quả cho thấy LLM lớn zero-shot có chi phí cao nhất, trong khi LLM nhỏ zero-shot có chi phí thấp nhất trong ba cấu hình. Tuy nhiên, xét đồng thời độ trễ và F1 trong lần thử nghiệm này, sử dụng LLM để xử lý toàn bộ tin nhắn chưa phải phương án phù hợp.

Một hướng triển khai có thể cân nhắc là sử dụng BoW + MLP làm mô hình phân loại chính để xử lý phần lớn tin nhắn. Những trường hợp khó hoặc có độ tin cậy thấp có thể được đưa sang một bước kiểm tra bổ sung, nếu hệ thống có đủ tài nguyên và vẫn bảo đảm yêu cầu về thời gian phản hồi.

Trước khi triển khai chính thức, cần benchmark lại trên hạ tầng production với lưu lượng tương đương khoảng 116 tin/giây. Kết quả đo trong notebook chưa đủ để bảo đảm hệ thống sẽ duy trì được SLA khi vận hành ở quy mô thực tế.

## 7. AI log và điều đã học

### Nhật ký sử dụng AI

| Nội dung | AI hỗ trợ | Phần em kiểm tra và chỉnh sửa |
|---|---|---|
| Phân tích kết quả mô hình | Hỗ trợ diễn giải sự khác biệt giữa BoW, MLP, MiniLM và LLM. | Đối chiếu F1, Accuracy và các chỉ số hiệu năng với kết quả thực nghiệm. |
| Phân tích dự đoán và thực tế | Hỗ trợ tổ chức các dự đoán thành bảng đối chiếu và giải thích sai lệch. | Chỉ giữ lại những kết quả có số liệu hoặc thông tin xác nhận trong báo cáo. |
| Tính số tham số MLP | Hỗ trợ giải thích công thức tính trọng số và bias của các lớp Linear. | Kiểm tra lại công thức với kiến trúc đầu vào 2.000 chiều và hai cấu hình hidden = 4, 512. |
| Phân tích 50 case khó | Hỗ trợ diễn giải sự khác biệt giữa các nhóm dữ liệu và mô hình. | Đối chiếu tỷ lệ dự đoán đúng theo từng nhóm case. |
| Lựa chọn mô hình triển khai | Hỗ trợ so sánh chất lượng phân loại, độ trễ và chi phí. | Căn cứ vào SLA p95 < 50 ms để lựa chọn mô hình phù hợp. |
| Hoàn thiện báo cáo | Hỗ trợ biên tập câu từ, sắp xếp nội dung và trình bày kết quả. | Kiểm tra số liệu và không sử dụng kết quả chưa được thực nghiệm xác nhận. |

### Điều em hiểu ra sau bài thực hành

Qua bài thực hành, em hiểu rõ hơn sự khác biệt giữa Bag-of-Words, MLP, Embedding và LLM trong bài toán phân loại tin nhắn rác. Mỗi phương pháp có ưu điểm và hạn chế riêng, không có mô hình nào luôn tốt nhất trong mọi trường hợp.

Em nhận ra rằng F1 cao trên tập test chưa đủ để quyết định lựa chọn mô hình triển khai. Độ trễ p95, tốc độ xử lý, thời gian huấn luyện và chi phí vận hành cũng là những yếu tố quan trọng. Một mô hình có chất lượng phân loại tốt nhưng không đáp ứng SLA vẫn có thể không phù hợp với yêu cầu của sản phẩm.

Ngoài ra, việc đánh giá trên 50 case khó giúp em nhận thấy các mô hình dựa trên từ khóa dễ gặp khó khăn với spam tự nhiên, nội dung thao túng và tin nhắn cố tình né từ khóa. Những trường hợp này đòi hỏi khả năng nắm bắt ngữ cảnh tốt hơn.

Cuối cùng, em hiểu rằng AI chỉ nên được sử dụng như một công cụ hỗ trợ. Mọi kết quả, công thức và kết luận trong báo cáo đều cần được kiểm tra lại bằng notebook và dữ liệu thực nghiệm. Việc kết hợp kiến thức lý thuyết với kiểm chứng thực tế giúp quá trình xây dựng và lựa chọn mô hình có cơ sở đáng tin cậy hơn.