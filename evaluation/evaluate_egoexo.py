
import json
import argparse
from pycocotools import mask as mask_utils
import numpy as np
import tqdm
from sklearn.metrics import balanced_accuracy_score
import cv2
import evaluation.utils as utils

CONF_THRESH = 0.5

def evaluate_take(gt, pred, take_id, get_iou_per_imageid=False):
    
    IoUs = []
    ShapeAcc = []
    ExistenceAcc = []
    LocationScores = []

    ObjExist_GT = []
    ObjExist_Pred = []

    ObjSizeGT = []
    ObjSizePred = []
    IMSize = []
    
    # Dictionary to store per-image IoU values
    per_image_ious = {} if get_iou_per_imageid else None

    for object_id in gt['masks'].keys():
        ego_cams = [x for x in gt['masks'][object_id].keys() if 'aria' in x]
        if len(ego_cams) < 1:
            continue
        assert len(ego_cams) == 1
        EGOCAM = ego_cams[0]

        EXOCAMS = [x for x in gt['masks'][object_id].keys() if 'aria' not in x]
        for exo_cam in EXOCAMS:
            gt_masks_ego = {}
            gt_masks_exo = {}
            pred_masks_exo = {}

            if EGOCAM in gt["masks"][object_id].keys():
                gt_masks_ego = gt["masks"][object_id][EGOCAM]
            if exo_cam in gt["masks"][object_id].keys():
                gt_masks_exo = gt["masks"][object_id][exo_cam]
            if object_id in pred["masks"].keys() and f'{EGOCAM}_{exo_cam}' in pred["masks"][object_id].keys():
                pred_masks_exo = pred["masks"][object_id][f'{EGOCAM}_{exo_cam}']

            for frame_idx in gt_masks_ego.keys():
                
                if int(frame_idx) not in gt["annotated_frames"][object_id][exo_cam]:
                    continue
                
                if not frame_idx in gt_masks_exo:
                    gt_mask = None
                    gt_obj_exists = 0
                else:
                    gt_mask = mask_utils.decode(gt_masks_exo[frame_idx])

                    gt_obj_exists = 1

                try:
                    pred_mask = mask_utils.decode(pred_masks_exo[frame_idx]["pred_mask"])
                except:
                    breakpoint()

                pred_obj_exists = int(pred_masks_exo[frame_idx]["confidence"] > CONF_THRESH)

                if gt_obj_exists:
                    # iou and shape accuracy
                    try:
                        # Change gt shape to match pred shape
                        if gt_mask.shape != pred_mask.shape:
                            gt_mask = cv2.resize(gt_mask, (pred_mask.shape[1], pred_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
                        iou, shape_acc = utils.eval_mask(gt_mask, pred_mask)
                    except:
                        breakpoint()

                    # compute existence acc i.e. if gt == pred == ALL ZEROS or gt == pred == SOME MASK
                    ex_acc = utils.existence_accuracy(gt_mask, pred_mask)

                    # # location accuracy
                    location_score = utils.location_score(gt_mask, pred_mask, size=(gt_mask.shape[0], gt_mask.shape[1]))

                    IoUs.append(iou)
                    ShapeAcc.append(shape_acc)
                    ExistenceAcc.append(ex_acc)
                    LocationScores.append(location_score)

                    ObjSizeGT.append(np.sum(gt_mask).item())
                    ObjSizePred.append(np.sum(pred_mask).item())
                    IMSize.append(list(gt_mask.shape[:2]))
                    
                    # Store per-image IoU if requested
                    if get_iou_per_imageid:
                        # Create image ID as "{take_id}_{camname}_{objectname}"
                        # For ego->exo, we use exo_cam as the target camera
                        image_id = f"{take_id}_{exo_cam}_{object_id}"
                        if image_id not in per_image_ious:
                            per_image_ious[image_id] = {}
                        per_image_ious[image_id][frame_idx] = iou

                ObjExist_GT.append(gt_obj_exists)
                ObjExist_Pred.append(pred_obj_exists)

    IoUs = np.array(IoUs)
    ShapeAcc = np.array(ShapeAcc)
    ExistenceAcc = np.array(ExistenceAcc)
    LocationScores = np.array(LocationScores)

    return IoUs.tolist(), ShapeAcc.tolist(), ExistenceAcc.tolist(), LocationScores.tolist(), \
            ObjExist_GT, ObjExist_Pred, ObjSizeGT, ObjSizePred, IMSize, per_image_ious

def validate_predictions(gt, preds):

    assert "ego-exo" in preds
    preds = preds["ego-exo"]

    assert type(preds) == type({})
    for key in ["results"]:
        assert key in preds.keys()

    assert len(preds["results"]) == len(gt["annotations"])
    for take_id in gt["annotations"]:
        assert take_id in preds["results"]

        for key in ["masks", "subsample_idx"]:
            assert key in preds["results"][take_id]

        # check objs
        assert len(preds["results"][take_id]["masks"]) == len(gt["annotations"][take_id]["masks"])
        for obj in gt["annotations"][take_id]["masks"]:
            assert obj in preds["results"][take_id]["masks"], f"{obj} not in pred {take_id}"

            ego_cam = None
            exo_cams = []
            for cam in gt["annotations"][take_id]["masks"][obj]:
                if 'aria' in cam:
                    ego_cam = cam
                else:
                    exo_cams.append(cam)
            try:
                assert not ego_cam is None
            except:
                continue
            try:
                assert len(exo_cams) > 0
            except:
                continue

            for cam in exo_cams:
                try:
                    assert f"{ego_cam}_{cam}" in preds["results"][take_id]["masks"][obj]
                except:
                    breakpoint()

                for idx in gt["annotations"][take_id]["masks"][obj][ego_cam]:
                    assert idx in preds["results"][take_id]["masks"][obj][f"{ego_cam}_{cam}"]

                    for key in ["pred_mask", "confidence"]:
                        assert key in preds["results"][take_id]["masks"][obj][f"{ego_cam}_{cam}"][idx]

def evaluate_egoexo(gt, preds, get_iou_per_imageid=False):

    validate_predictions(gt, preds)
    preds = preds["ego-exo"]

    total_iou = []
    total_shape_acc = []
    total_existence_acc = []
    total_location_scores = []

    total_obj_sizes_gt = []
    total_obj_sizes_pred = []
    total_img_sizes = []

    total_obj_exists_gt = []
    total_obj_exists_pred = []
    
    # Dictionary to store per-image IoU values
    iou_per_imageid = {} if get_iou_per_imageid else None

    for take_id in tqdm.tqdm(gt["annotations"]):

        ious, shape_accs, existence_accs, location_scores, obj_exist_gt, obj_exist_pred, \
            obj_size_gt, obj_size_pred, img_sizes, per_image_ious = evaluate_take(gt["annotations"][take_id], 
                                                                  preds["results"][take_id], take_id, get_iou_per_imageid)

        total_iou += ious
        total_shape_acc += shape_accs
        total_existence_acc += existence_accs
        total_location_scores += location_scores

        total_obj_sizes_gt += obj_size_gt
        total_obj_sizes_pred += obj_size_pred
        total_img_sizes += img_sizes

        total_obj_exists_gt += obj_exist_gt
        total_obj_exists_pred += obj_exist_pred
        
        # Collect per-image IoU data
        if get_iou_per_imageid and per_image_ious:
            iou_per_imageid.update(per_image_ious)
    
    balanced_acc = balanced_accuracy_score(total_obj_exists_gt, total_obj_exists_pred)
    print('TOTAL EXISTENCE BALANCED ACC: ', balanced_acc)
    iou = np.mean(total_iou)
    print('TOTAL IOU: ', iou)
    location_score = np.mean(total_location_scores)
    print('TOTAL LOCATION SCORE: ', location_score)
    shape_acc = np.mean(total_shape_acc)
    print('TOTAL SHAPE ACC: ', shape_acc)
    result_dict = {'iou': iou, 'shape_acc': shape_acc, 'location_score': location_score, 'balanced_acc': balanced_acc}
    
    # Add per-image IoU data if requested
    if get_iou_per_imageid:
        result_dict['iou_per_imageid'] = iou_per_imageid
    
    return result_dict

def main(args):

    # load gt and pred jsons
    with open(args.gt_file, 'r') as fp:
        gt = json.load(fp)

    with open(args.pred_file, 'r') as fp:
        preds = json.load(fp)

    # evaluate
    evaluate_egoexo(gt, preds)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt-file', type=str, required=True, 
                            help="path to json with gt annotations")
    parser.add_argument('--pred-file', type=str, required=True,
                            help="")
    args = parser.parse_args()
    
    main(args)