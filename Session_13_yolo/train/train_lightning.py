import configparser
import ast
import torch

from model1 import YOLOv3
from yolo_loss import YoloLoss
from pytorch_lightning import LightningModule, Trainer
from utils1 import (mean_average_precision,cells_to_bboxes,get_evaluation_bboxes,
            save_checkpoint,load_checkpoint,check_class_accuracy,get_loaders,plot_couple_examples)
from torch.optim.lr_scheduler import OneCycleLR

loss_fn = YoloLoss()


class LitYOLOv3(LightningModule):
    def __init__(self,config_file = r"C:\Users\vmt8kor\Desktop\vinoth_documents\vinoth_documents1\python_files\School_AI\Session_13_yolo\config\yolo_config.cfg"):
        super().__init__()
        self.config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
        self.config.read(config_file)

        self.device1 = self.config['DEFAULT']['DEVICE']
        self.num_workers = int(self.config['DEFAULT']['NUM_WORKERS'])
        self.batch_size = int(self.config['DEFAULT']['BATCH_SIZE'])
        self.image_size = int(self.config['DEFAULT']['IMAGE_SIZE'])
        self.num_classes = int(self.config['DEFAULT']['NUM_CLASSES'])
        self.learning_rate = float(self.config['DEFAULT']['LEARNING_RATE'])
        self.weight_decay = float(self.config['DEFAULT']['WEIGHT_DECAY'])
        self.num_epochs = int(self.config['DEFAULT']['NUM_EPOCHS'])
        self.conf_threshold = float(self.config['DEFAULT']['CONF_THRESHOLD'])
        self.map_iou_thresh = float(self.config['DEFAULT']['MAP_IOU_THRESH'])
        self.nms_iou_thresh = float(self.config['DEFAULT']['NMS_IOU_THRESH'])
        self.scale_size = ast.literal_eval(self.config['DEFAULT']['SCALE_SIZE'])
        self.pin_memory = self.config['DEFAULT']['PIN_MEMORY']
        self.load_model = self.config['DEFAULT']['LOAD_MODEL']
        self.save_model = self.config['DEFAULT']['SAVE_MODEL']
        self.checkpoint_file = self.config['DEFAULT']['CHECKPOINT_FILE']
        self.train_csv_path = self.config['DEFAULT']['TRAIN_CSV_PATH']
        self.test_csv_path = self.config['DEFAULT']['TEST_CSV_PATH']
        self.img_dir = self.config['DEFAULT']['IMG_DIR']
        self.label_dir = self.config['DEFAULT']['LABEL_DIR']
        self.anchors = ast.literal_eval(self.config['DEFAULT']['ANCHORS'])
        self.means = ast.literal_eval(self.config['DEFAULT']['means'])

        self.example_input_array = torch.Tensor(self.batch_size, 3, self.image_size, self.image_size)

        self.model = YOLOv3(num_classes=self.num_classes)
        self.save_hyperparameters()
        # self.lr = config.LEARNING_RATE

    def forward(self, imgs):
        detections = self.model(imgs)
        return detections

    def criterion(self, out, y):
        y0, y1, y2 = (y[0], y[1], y[2])
        self.scale_size = [y[i].shape[2] for i in range(len(y))][::-1]
        scaled_anchors = (torch.tensor(self.anchors)
                * torch.tensor(self.scale_size).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)).to(self.device1)
        loss = (loss_fn(out[0], y0, scaled_anchors[0])
                + loss_fn(out[1], y1, scaled_anchors[1])
                + loss_fn(out[2], y2, scaled_anchors[2]))
        return loss

    def training_step(self, batch, batch_id):
        x, y = batch
        out = self(x)
        loss = self.criterion(out, y)
        self.log("training loss", loss, prog_bar=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=True)
        return loss

    def on_train_end(self) -> None:
        scaled_anchors = (torch.tensor(self.anchors)
                * torch.tensor(self.scale_size).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)).to(self.device1)

        plot_couple_examples(self.model, self.test_dataloader(), 0.6, 0.5, scaled_anchors)

        check_class_accuracy(self.model, self.train_dataloader(), threshold=self.conf_threshold)
        pred_boxes, true_boxes = get_evaluation_bboxes(
            self.test_dataloader(),
            self.model,
            iou_threshold=self.nms_iou_thresh,
            anchors=self.anchors,
            threshold=self.conf_threshold,
        )
        mapval = mean_average_precision(
            pred_boxes,
            true_boxes,
            iou_threshold=self.map_iou_thresh,
            box_format="midpoint",
            num_classes=self.num_classes,
        )
        print(f"MAP: {mapval.item()}")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            # momentum=0.9,
            # weight_decay=5e-4,
        )
        EPOCHS = self.num_epochs * 2 // 5

        scheduler_dict = {
            "scheduler": OneCycleLR(
                optimizer,
                max_lr=1E-3,
                steps_per_epoch=len(self.train_dataloader()),
                epochs=EPOCHS,
                pct_start=5 / EPOCHS,
                div_factor=100,
                three_phase=False,
                final_div_factor=100,
                anneal_strategy='linear'
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}

    def setup(self, stage=None):
        self.train_loader, self.test_loader, self.train_eval_loader = get_loaders(self.train_csv_path,self.test_csv_path,
                            self.batch_size,self.img_dir,self.label_dir,self.anchors,self.scale_size,self.num_workers,self.pin_memory)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.train_eval_loader

    def test_dataloader(self):
        return self.test_loader

if __name__ == "__main__":
    import torch

    torch.cuda.empty_cache()