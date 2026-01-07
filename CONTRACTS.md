# HỢP ĐỒNG DỮ LIỆU (DATA CONTRACTS) V1.0

**Người chịu trách nhiệm:** Lại Trí Dũng
**Ngày phát hành:** 17/11/2025
**Trạng thái:** CHÍNH THỨC (V1.0)

## Tóm tắt Nguồn dữ liệu

* **Nguồn:** `data_raw` (11,508 ảnh đã làm sạch).
* **Kích thước đồng nhất:** Tất cả ảnh đều là 1280x720.
* **Timestamp:** Không có (Hệ thống chỉ hỗ trợ xử lý stateless - phi trạng thái).


## 1. KAFKA: Nguồn Dữ liệu Kích hoạt (Trigger)

**Topic:** `traffic-metadata`
**Mô tả:** Luồng metadata ảnh thô. P2 (Data Engineer) push lên sau khi ảnh đã được đóng gói vào LMDB. Đây là "Hợp đồng Đầu vào" (Input Contract) cho P1 (Spark).

### Cấu trúc Message (JSON Value):

```json
{
  "record_key": "cam_01_00544",
  "camera_id": "cam_01",
  "lmdb_info": {
    "lmdb_filepath": "/traffic-data/2025-11-17.lmdb",
    "frame_height": 720,
    "frame_width": 1280
  },
  "schema_version": "1.0"
}
```
**Chi tiết từng trường (Kafka):**

| Trường | Kiểu dữ liệu | Bắt buộc? | Mô tả |
| :--- | :--- | :--- | :--- |
| **`record_key`** | String | Có | Lấy từ tên file sạch (ví dụ: "cam\_01\_00544"). Sẽ dùng làm \_id cho MongoDB. |
| **`camera_id`** | String | Có | Trích xuất từ tên file (ví dụ: "cam\_01", "cam\_11"). |
| **`lmdb_info`** | Object | Có | Object chứa thông tin về nơi lấy ảnh thô. |
| `lmdb_info.lmdb_filepath` | String | Có | Đường dẫn file LMDB trên MinIO (do P2 tạo ra). |
| `lmdb_info.frame_height` | Integer | Có | (Đã chốt) Luôn luôn là 720. |
| `lmdb_info.frame_width` | Integer | Có | (Đã chốt) Luôn luôn là 1280. |
| **`schema_version`** | String | Có | Phiên bản 1.0, dùng để quản lý thay đổi sau này. |

## 2. MONGODB: Nơi lưu trữ Kết quả (Serving Layer)

**Database:** `traffic_db`
**Collection:** `analysis_results`
**Mô tả:** Nơi lưu trữ kết quả sau khi P1 (Spark) chạy YOLO inference. Đây là "Hợp đồng Đầu ra" (Output Contract) cho P5 (BI Analyst).

### Cấu trúc Document (JSON):

```json
{
  "_id": "cam_01_00544",
  "processing_timestamp": ISODate("2025-11-17T12:30:02.500Z"),
  "camera_id": "cam_01",
  "frame_dimensions": {
    "width": 1280,
    "height": 720
  },
  "traffic_status": "Free",
  "inference_metrics": {
    "counts": {
      "total": 12,
      "car": 7,
      "motorbike": 4,
      "bus": 0,
      "truck": 1
    },
    "density_percent": 0.25
  },
  "debug_pixels": {
    "road_pixels": 600500,
    "vehicle_pixels": 50000
  },
  "raw_yolo_results": [
    {"class_name": "car", "confidence": 0.92, "box_xywh": [100, 150, 50, 80]},
    {"class_name": "motorbike", "confidence": 0.88, "box_xywh": [120, 170, 30, 60]}
  ],
  "pipeline_info": {
    "kafka_schema_version": "1.0",
    "model_checkpoint": "yolov8m_traffic_v1.2.pt",
    "processing_time_ms": 2377
  }
}
```

**Chi tiết từng trường (MongoDB):**

| Trường | Kiểu dữ liệu | Mô tả |
| :--- | :--- | :--- |
| **`_id`** | String | Lấy từ `record_key` của Kafka. |
| **`processing_timestamp`** | ISODate | Do Spark tạo ra (current\_timestamp()). Giúp P5 lọc "kết quả mới nhất". |
| **`camera_id`** | String | Lấy từ Kafka. |
| **`frame_dimensions`** | Object | (Đã chốt) Luôn là 1280x720. |
| **`traffic_status`** | String | Trạng thái giao thông ("Free", "Normal", "Congested"). |
| **`inference_metrics`** | Object | Kết quả cốt lõi (được P1/P4 tính toán bên trong Spark). |
| `inference_metrics.counts` | Object | Số lượng đếm được cho từng loại (class) phương tiện từ YOLO. |
| `inference_metrics.density_percent` | Double | Tỷ lệ mật độ (ví dụ: tổng diện tích box / diện tích ảnh). |
| `debug_pixels` | Object | Số lượng pixel dùng để debug công thức tính mật độ. |
| **`raw_yolo_results`** | Array[Object] | Mảng chứa output thô từ mô hình YOLO, hữu ích cho P4 debug. |
| **`pipeline_info`** | Object | Metadata về quá trình chạy (dùng để gỡ lỗi). |
| `pipeline_info.model_checkpoint` | String | Tên file model checkpoint (ví dụ: v1.2.pt) mà P4 cung cấp. |
| `pipeline_info.processing_time_ms` | Integer | Thời gian (ms) Spark dùng để xử lý (từ lúc đọc Kafka đến lúc ghi Mongo). |