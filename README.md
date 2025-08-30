# Plastic-Object-Detection-DINOv2-style-SSL-using-YOLO-backbone
Plastic detection using DINOv2-style Self-Supervised Learning pretraining YOLOv10, YOLOv11, and YOLOv12 backbone, followed by fine-tuned YOLO detector

# Summary

This notebook implements a **DINOv2-style Self-Supervised Learning (SSL)** pretraining on the **YOLOv10, YOLOv11, and YOLOv12 backbone** (conv net), followed by fine-tuning a YOLOv10, YOLOv11, and YOLOv12 detector on Plastic object besides seashore dataset. Finally, it visualizes learned features via 2D **PCA** with k-means coloring.

# Description:
   1) COCO → YOLO conversion (images/labels + data.yaml).
   2) SSL pretraining (DINOv2-style):
       - Student/Teacher (EMA) with cosine momentum schedule
       - Multi-crop views: 2 global + 8 local
       - Temperature schedule for teacher + probability centering
       - Cross-entropy between teacher probs (global) and student logits (all views)
   3) Save ONLY the SSL-pretrained YOLOv10 backbone weights.
   4) Fine-tune YOLOv10 detector initialized from those weights.
   5) Evaluate (mP, mR, mAP@0.50, mAP@0.50–0.95).
   6) PCA of backbone features (unsupervised clusters).

Here are the Kaggle link of Plastic Detection DINOv2-style SSL using YOLOv10, YOLOv11, and YOLOv12 backbone: 
https://www.kaggle.com/code/neelunzn/plastic-detection-dinov2-style-self-sl-yolov10s 
https://www.kaggle.com/code/neelunzn/plastic-detection-dinov2-style-self-sl-yolov11s
https://www.kaggle.com/code/neelunzn/plastic-detection-dinov2-style-self-sl-yolov12s
