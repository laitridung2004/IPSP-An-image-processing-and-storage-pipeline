# BIG DATA STORAGE AND PROCESSING PIPELINE
## IPSP: An Image Processing and Storage Pipeline

![image](https://github.com/user-attachments/assets/9010e61d-2818-4a8a-b92c-28bd567251e8)
***

### 1. Introduction

In the era of smart cities, real-time traffic analysis is crucial for managing urban congestion and enhancing public safety. Accurately monitoring vehicle flow, density, and classification is a significant Big Data challenge.

This project represents our team's initiative to design and implement a highly scalable, low-latency, end-to-end data pipeline. By simulating a feed of 11,508 competition images, we built a robust, cloud-native system capable of ingesting, processing, and visualizing traffic data in near real-time.

- **Dataset:** The "11,508 competition images" dataset used for this project can be accessed here: [Google Drive](https://drive.google.com/drive/u/1/folders/1zlLvF1dJn7C29yZDsCrdOs1clVcYLSiC)

Our approach integrates a modern Big Data stack, including **Apache Kafka** for event streaming, **PySpark Structured Streaming** for distributed processing, and **YOLO** for real-time AI inference. The entire system is containerized with **Docker** and orchestrated using **Kubernetes (K8s)**, demonstrating a complete Kappa-like architecture.

***This project is intended for `educational purposes only` for the Big Data Storage and Pipeline course. The model's performance is subject to dataset limitations and does not guarantee 100% accurate prediction.***
***

### 2. Technology Stack

| Category | Technology | Role |
| :--- | :--- | :--- |
| **Orchestration** | **Kubernetes (K8s)** | Manages and orchestrates all containerized services. |
| **Containerization** | **Docker** | Packages each application (Producer, Spark, etc.). |
| **Stream Ingestion** | **Apache Kafka** | Decouples services with a reliable, persistent event log. |
| **Stream Processing** | **PySpark Streaming** | Core engine for consuming, transforming, and running AI inference. |
| **Batch Preprocessing**| **PySpark Batch** | Optimizes 11,508 images into LMDB format. |
| **Storage (Source)** | **LMDB** | Efficient Key-Value store for binary image data. |
| **Storage (Object)** | **MinIO** | S3-compatible storage for LMDB files and AI models. |
| **Serving Layer DB** | **MongoDB** | NoSQL database to store final analysis results. |
| **Visualization** | **Grafana** | Real-time dashboard for monitoring traffic metrics. |
| **AI/ML** | **YOLOv8** / **Ultralytics** | Object detection model (trained offline, run online). |

***

### 3. Key Features

- **Scalable Preprocessing:** Utilized **PySpark Batch** to process 11,508 images into the **LMDB** (Lightning Memory-Mapped Database) format, optimizing I/O for fast retrieval.

- **Resilient Data Ingestion:** Implemented an **Apache Kafka** producer to publish image metadata, creating a fault-tolerant "trigger" for the processing pipeline.

- **Distributed Stream Processing:** Built the core pipeline with **PySpark Structured Streaming** to consume metadata from Kafka in micro-batches.

- **Real-time AI Inference:** Integrated a pre-trained **YOLO** model directly into the Spark pipeline using Pandas UDFs, allowing for distributed AI inference on-the-fly.

- **Efficient Data Fetching:** Spark jobs fetch image data directly from **MinIO** (using LMDB) based on the Kafka trigger, separating metadata from heavy data.

- **High-Performance Serving:** Stored structured analysis results (vehicle counts, density) in **MongoDB**, optimized for fast writes and analytical queries from the dashboard.

- **Real-time Visualization:** Connected **Grafana** to MongoDB to display live traffic metrics, including vehicle counts per camera and density heatmaps.

- **Cloud-Native Deployment:** Containerized all services (Kafka, Spark, MinIO, etc.) and deployed them on **Kubernetes** for automated scaling and self-healing.

***

### 4. Project Structure

- **.gitignore**:  Configuration file to exclude data directories (e.g., data_clean/), virtual environments (.venv/), and compiled output files (*.lmdb, *.pt) from Git version control.

- **CONTRACTS.md**: The official Data Contract (V1.0) document. It defines the data schemas for the Kafka topic and the MongoDB collection, serving as the single source of truth for **Spark Architecture**, **Data Engineer**, and **BI & Data Analyst**.

- **README.md**: The main project documentation, providing a comprehensive overview, architectural diagrams, technology stack, and setup/deployment instructions.

- **0_pipelines/**: Holds high-level project documentation, including the core architecture diagrams, task assignments, and sprint planning artifacts.

- **1_infrastructure/**: **DevOps Engineer**'s (DevOps) directory, containing all .yaml manifest files for deploying the full infrastructure stack (MinIO, Kafka, MongoDB) on Kubernetes, along with the Dockerfiles for application containerization.

- **2_preprocessing/**: **Data Engineer**'s directory, which includes preprocess_lmdb.py (the Spark Batch Job for creating the LMDB) and kafka_producer.py (the script used to ingest 11k metadata records into Kafka).

- **3_spark_processor/**: The core of the project (**Spark Architect**'s directory). This contains stream_processor.py, the main PySpark Structured Streaming application responsible for consuming from Kafka, executing the AI model, and writing results to MongoDB.

- **4_model_training/**: **Data Scientist**'s directory. Contains train.py (for local YOLO model training), inference.py (the inference logic for **Spark Architecture** to integrate), and the models/ directory (which holds .pt files before **Data Scientist** uploads them to MinIO).

***

### 5. Acknowledgments
This project would not have been possible without the invaluable contributions of the open-source community, particularly the teams behind **Apache Spark**, **Apache Kafka**, and **Kubernetes**.

We extend our heartfelt gratitude to our lecturers, **Tran Viet Trung**, for assigning us this challenging yet captivating project. It has been an incredible learning opportunity that has significantly enhanced our knowledge and skillset in Big Data engineering.

Our sincere thanks also go to our professors in the Department for their unwavering support and sharing during the whole course.

***

### 6. Contributors

- **Spark Architect:** Lai Tri Dung - 20225486
- **Data Engineer:** Nguyễn Trọng Tâm - 20225527
- **DevOps Engineer:** Vũ Hữu An - 20225497
- **Data Scientist:** Lưu Thiện Việt Cường - 20225477
- **BI & Data Analyst:** Đàm Quang Đức - 20225483
***

### 7. License
This project is licensed under the [MIT License](LICENSE).




