import torch


def DiceLoss(y_true, y_pred, epsilion=1e-15):
  def dice_coef(y_true, y_pred, epsilion):
    intersection = torch.sum(y_true * y_pred, dim=(0, 2, 3))
    union = torch.sum(y_true, dim=(0, 2, 3)) + torch.sum(y_pred, dim=(0, 2, 3))
    return (2 * intersection + epsilion) / (union + epsilion)
  return 1 - dice_coef(y_true, y_pred, epsilion=epsilion)


def IOU(y_pred, y_true, num_classes, threshold=0.5, epsilon=1e-7):
    assert y_pred.shape == y_true.shape
    num_classes = y_pred.shape[1]
    y_pred = (y_pred > threshold).float()
    intersection = torch.sum(y_pred * y_true, dim=(0, 2, 3))  # Sum over batch, height, width
    union = torch.sum(y_pred, dim=(0, 2, 3)) + torch.sum(y_true, dim=(0, 2, 3)) - intersection
    iou_per_class = (intersection + epsilon) / (union + epsilon)
    return iou_per_class.mean()

def FocalLoss(y_true, y_pred, epsilon=1e-7):
  alpha = 0.25
  gamma = 2

  def focal_loss_with_logits(logits, targets, alpha, gamma, y_pred):
    weight_a = alpha * (1 - y_pred) ** gamma * targets
    weight_b = (1 - alpha) * y_pred ** gamma * (1 - targets)
    return (torch.log1p(torch.exp(-torch.abs(logits))) + 
            torch.nn.functional.relu(-logits)) * (weight_a + weight_b) + logits * weight_b
    
  y_pred = torch.clamp(y_pred, epsilon, 1.0 - epsilon)
  logits = torch.log(y_pred / (1 - y_pred))
  loss = focal_loss_with_logits(logits = logits, targets = y_true, alpha = alpha, gamma = gamma, y_pred = y_pred)
  return torch.mean(loss)
