# Logic model HPE-Li DSK-only 3D

## 1. Mục tiêu

Repository này là thí nghiệm ablation để trả lời:

> Khi giữ nguyên backbone, dữ liệu, cấu hình huấn luyện và regression head,
> việc bỏ Channel Transformer ảnh hưởng thế nào đến kết quả pose 3D?

Đây không phải bản sao tuyệt đối của cấu hình HPE-Li ECCV 2024. Model sử dụng
logic DSKConv không Transformer của HPE-Li, nhưng giữ kích thước backbone của
HPE-Li++ để so sánh công bằng.

## 2. Cặp mô hình so sánh

```text
HPE-Li-3D / HPE-Li++
  DSKConv++ + ChannelTransformer

Demo HPE-Li 1
  DSKConv, không ChannelTransformer
```

Các thành phần khác phải giống nhau:

| Thành phần | Giá trị |
|---|---:|
| Input | `(B, 3, 114, 10)` |
| Output | `(B, 17, 3)` |
| Base channel | `128` |
| Stage 2 channel | `256` |
| Số DSK branch | `3` |
| Convolution groups | `32` |
| Reduction ratio | `4` |
| Regression input | `3584` |
| Regression output | `51` |

## 3. Luồng tổng thể

```text
CSI (B, 3, 114, 10)
  ->
SKUnit 1
  ->
BatchNorm
  ->
SKUnit 2
  ->
AveragePool 2x2
  ->
RegressionHead
  ->
Pose (B, 17, 3)
```

Shape:

| Giai đoạn | Shape |
|---|---:|
| Input | `(B, 3, 114, 10)` |
| SKUnit 1 | `(B, 128, 57, 5)` |
| SKUnit 2 | `(B, 256, 28, 2)` |
| Final pool | `(B, 256, 14, 1)` |
| Flatten | `(B, 3584)` |
| Regression | `(B, 51)` |
| Output | `(B, 17, 3)` |

## 4. SKUnit

Mỗi `SKUnit` thực hiện:

```text
Conv 1x1
  ->
BatchNorm + ReLU
  ->
AveragePool 2x2
  ->
DSKConv
  ->
BatchNorm
  ->
Conv 1x1 + BatchNorm
```

Hai model ablation và HPE-Li++ dùng cùng cấu trúc `SKUnit`.

## 5. DSKConv

Đầu vào:

```text
x: (B, C, F, T)
```

Ba grouped convolution chạy song song với dilation `1`, `2`, `3`:

```text
U_k1, U_k2, U_k3: (B, C, F, T)
```

Sau khi stack:

```text
U: (B, M, C, F, T)
```

với `M = 3`.

### 5.1. Channel-wise selective kernel attention

CwSKA trả lời:

> Với từng feature channel, kernel branch nào nên được ưu tiên?

```text
U
  -> sum theo branch
  -> global average pool trên F và T
  -> bottleneck FC
  -> sinh M bộ trọng số channel
  -> softmax theo M
  -> weighted sum các branch
```

Kết quả:

```text
V_channel: (B, C, F, T)
```

### 5.2. Frequency-wise selective kernel attention

FwSKA trả lời:

> Với từng hàng subcarrier, kernel branch nào nên được ưu tiên?

```text
U
  -> sum theo channel C
  -> average theo time T
  -> softmax theo M
  -> weighted sum các branch
```

Kết quả:

```text
V_frequency: (B, C, F, T)
```

### 5.3. Fusion không Transformer

Hai feature được concat theo chiều thời gian:

```text
(B, C, F, T) + (B, C, F, T)
  ->
(B, C, F, 2T)
```

Sau đó:

```text
BatchNorm
  ->
AveragePool kernel (1, 2)
  ->
(B, C, F, T)
```

Đây là điểm ablation:

```text
HPE-Li++:
  concat -> BatchNorm -> ChannelTransformer -> AveragePool

DSK-only:
  concat -> BatchNorm -> AveragePool
```

Không thay phép concat bằng phép cộng vì việc đó sẽ thay đổi thêm một yếu tố
ngoài Transformer.

### 5.4. Kiểm chứng số tham số

Với cấu hình mặc định:

```text
HPE-Li++:  2,056,851 parameters
DSK-only:    393,363 parameters
Chênh lệch: 1,663,488 parameters
```

HPE-Li++ có đúng `1,663,488` tham số nằm trong các module Transformer. Vì vậy
chênh lệch tổng tham số bằng chính xác số tham số Transformer bị loại bỏ.

Ngoài các key chứa `.transformer.`, 120 state-dict key còn lại của hai model
trùng nhau. Đây là kiểm chứng trực tiếp rằng convolution branch, CwSKA, FwSKA,
BatchNorm, SKUnit và regression head vẫn giữ cùng cấu trúc tham số.

## 6. Regression head

Feature cuối:

```text
(B, 256, 14, 1)
```

được flatten và đưa qua:

```text
3584 -> 32 -> 64 -> 51
```

Sau đó reshape:

```text
(B, 51) -> (B, 17, 3)
```

## 7. Điều kiện để kết quả ablation hợp lệ

Hai lần train HPE-Li++ và DSK-only cần dùng cùng:

- dataset root;
- config split;
- random seed;
- batch size;
- số epoch;
- optimizer và learning rate;
- loss;
- pose normalization;
- early stopping;
- giới hạn batch;
- metric implementation.

Nên chạy nhiều seed và báo cáo trung bình cùng độ lệch chuẩn. Một lần chạy
đơn lẻ chưa đủ để kết luận mọi chênh lệch đến từ Transformer.

## 8. Kiểm tra trong code

Test `test_model_contains_no_transformer_modules_or_parameters` bảo đảm:

- không có module mang tên Transformer;
- state dict không có key Transformer;
- model vẫn tạo output `(B, 17, 3)`;
- feature trước regression vẫn là `(B, 256, 14, 1)`.

Vì vậy khác biệt kiến trúc có chủ đích chỉ nằm ở bước Transformer refinement.
