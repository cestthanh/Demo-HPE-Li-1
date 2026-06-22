# Logic model DSKNet 3D không Transformer

## 1. Lỗi đã gặp

Trước đó tôi đã chọn nhầm nguồn triển khai.

Tôi bám theo `SK_network.py` cũ, nơi dùng một kiểu SKConv lịch sử:

```text
cat -> view -> một attention freq-chan -> một output
```

Nhưng yêu cầu đúng là DSKConv trong `model/sknet_trans_mmfi.py`, dù class trong
file đó được đặt tên là `SKConv`. Forward đúng có:

```text
torch.stack
channel attention
frequency attention
concat hai output
BatchNorm
AvgPool2d(1, 2)
```

Phần `ChannelTransformer` trong file fork được loại bỏ cho baseline này.

## 2. Kiến trúc tổng thể

```text
CSI (B, 3, 114, 10)
  -> DSKUnit 1
       Conv1x1: 3 -> 128
       AvgPool2d(2,2)
       DSKConv no Transformer
       BatchNorm
       Conv1x1: 128 -> 128
  -> (B, 128, 57, 5)
  -> BatchNorm
  -> DSKUnit 2
       Conv1x1: 128 -> 256
       AvgPool2d(2,2)
       DSKConv no Transformer
       BatchNorm
       Conv1x1: 256 -> 256
  -> (B, 256, 28, 2)
  -> AvgPool2d(2,2)
  -> (B, 256, 14, 1)
  -> Flatten 3584
  -> Regression 3584 -> 32 -> 64 -> 51
  -> Pose (B, 17, 3)
```

## 3. Config model

```python
{
    "num_lay": 128,
    "hidden_reg": 32,
    "sk_m": 3,
    "sk_g": 32,
    "sk_r": 4,
    "sk_l": 32,
}
```

Ý nghĩa:

| Thành phần | Giá trị |
|---|---:|
| Base channel | `128` |
| Stage 2 channel | `256` |
| Số nhánh DSKConv | `3` |
| Conv groups | `32` |
| Reduction ratio | `4` |
| Bottleneck min | `32` |
| Transformer | Không |

## 4. DSKConv

Đầu vào:

```text
x: (B, C, H, W)
```

Ba nhánh convolution song song:

```text
branch 1: dilation=1, padding=1
branch 2: dilation=2, padding=2
branch 3: dilation=3, padding=3
```

Mỗi nhánh:

```text
Conv2d groups=32
BatchNorm2d
ReLU
```

Sau đó stack:

```python
feats = torch.stack([conv(x) for conv in self.convs], dim=1)
```

Shape:

```text
feats: (B, M, C, H, W)
```

## 5. Channel-wise selective kernel attention

Mục tiêu: với từng feature channel, chọn nhánh dilation phù hợp.

```text
feats
  -> sum theo M
  -> global average pool trên H,W
  -> Conv2d bottleneck
  -> M Conv2d sinh weight
  -> softmax theo M
  -> weighted sum theo M
```

Output:

```text
feats_channel: (B, C, H, W)
```

## 6. Frequency-wise selective kernel attention

Mục tiêu: với từng hàng frequency/subcarrier, chọn nhánh dilation phù hợp.

```text
feats
  -> sum theo C
  -> average pool theo W, giữ H
  -> softmax theo M
  -> weighted sum theo M
```

Output:

```text
feats_frequency: (B, C, H, W)
```

## 7. Fusion không Transformer

Source fork có:

```text
concat -> BatchNorm -> ChannelTransformer -> AvgPool2d(1,2)
```

Repo này dùng:

```text
concat -> BatchNorm -> AvgPool2d(1,2)
```

Shape:

```text
(B, C, H, W) + (B, C, H, W)
  -> concat theo W
(B, C, H, 2W)
  -> AvgPool2d(1,2)
(B, C, H, W)
```

Đây là khác biệt duy nhất trong DSKConv so với đoạn code bạn đưa: bỏ `self.tf`.

## 8. Training profile

Training vẫn theo Phase C Demo 2:

- P1-S1;
- random split ratio `0.8`, seed `0`;
- val/test split theo sequence với seed `41`;
- normalized MSE theo train XYZ mean/std;
- Adam, learning rate `0.001`, weight decay `0`;
- không scheduler;
- gradient clipping `1.0`;
- train cố định `50` epoch;
- không dùng early stopping;
- best checkpoint theo validation MPJPE;
- đánh giá test cuối bằng best checkpoint.

## 9. Kết luận

Bản đúng hiện tại không phải SKConv lịch sử `64/128`.

Bản đúng hiện tại là:

```text
DSKNet 3D no Transformer
  -> DSKConv dual CwSKA/FwSKA
  -> base channel 128
  -> grouped conv G=32
  -> output (B, 17, 3)
```
