# Layer-wise Relevance Propagation

## Bài toán phân lớp

Xem bài toán phân lớp, một bộ phân lớp (classifier) $f$ là một ánh xạ (mapping) $f: X -> \mathbb{R}$ mà $f(x) > 0$ thể hiện sự hiện diện của lớp


## Layer-wise Relevance Propagation (lan truyền mức độ liên quan khôn ngoan) for Neural Networks

Xem xét mạng neural gồm nhiều tầng của các neurons

$$
x_j = g\left(\sum_iw_{ij}x_i + b\right)
$$

Cho trước một ảnh $x$ và một bộ phân lớp $f$

Mục tiêu của lan truyền mức độ liên quan khôn ngoan là gán cho mỗi pixel điểm ảnh $p$ của $x$ một điểm số mức độ liên quan khôn ngoan (pixel-wise relevance score) $R_p^{(1)}$ mà

$$
f(x) \approx \sum_pR_p^{(1)}
$$