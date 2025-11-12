from csv import DictWriter
import os
from util.evaluations.metrics import compute_average_results

def write_csv(args, results):
    # return
    avg = compute_average_results(results)
    
    backbone = args.backbone
    save_dir = "evaluation_results"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    save_path = os.path.join(save_dir, f"{args.in_dataset}-{backbone}-{args.score}.csv")
    ood = args.out_datasets
    
    
    field_names = ['Methods']
    for o in ood:
        field_names.append(o+'-FPR')
        field_names.append(o+'-AUROC')
    # optional extra columns when forgetting is configured
    # we no longer append forget into results; read from args if available
    has_forget = False
    retain_acc = getattr(args, 'retain_acc', None)
    forget_fpr = getattr(args, 'forget_fpr', None)
    forget_auroc = getattr(args, 'forget_auroc', None)
    if (forget_fpr is not None) or (forget_auroc is not None):
        has_forget = True

    field_names.append('AVG-FPR')
    field_names.append('AVG-AUROC')
    # 增量指标（若存在）
    inc_overall = getattr(args, 'inc_overall_acc', None)
    inc_new = getattr(args, 'inc_new_acc', None)
    inc_old = getattr(args, 'inc_old_acc', None)
    if any(x is not None for x in [inc_overall, inc_new, inc_old]):
        field_names.append('Inc-Overall')
        field_names.append('Inc-New')
        field_names.append('Inc-Old')
        # 曲线统计（Final / Average）
        field_names.append('Inc-Final')
        field_names.append('Inc-Average')
    # 遗忘评估（仅非增量模式使用）
    if has_forget:
        field_names.append('forget-FPR')
        field_names.append('forget-AUROC')
    if retain_acc is not None:
        field_names.append('Retain-Acc')
    
    
    dict = {}
    dict["Methods"] = args.method
    for result, dataset in zip(results, ood):
        dict[f"{dataset}-FPR"] = 100.*result['FPR']
        dict[f"{dataset}-AUROC"] = 100.*result['AUROC']
        
    dict[f"AVG-FPR"] = 100.*avg['FPR']
    dict[f"AVG-AUROC"] = 100.*avg['AUROC']

    # 增量指标写出（若存在）
    if any(x is not None for x in [inc_overall, inc_new, inc_old]):
        try:
            if inc_overall is not None:
                dict['Inc-Overall'] = 100.*float(inc_overall)
            if inc_new is not None:
                dict['Inc-New'] = 100.*float(inc_new)
            if inc_old is not None:
                dict['Inc-Old'] = 100.*float(inc_old)
            # 同步 Final / Average（若已由评估打印并缓存曲线，可缺省；这里尝试从 args 读）
            inc_final = getattr(args, 'inc_final', None)
            inc_avg = getattr(args, 'inc_average', None)
            if inc_final is not None:
                dict['Inc-Final'] = 100.*float(inc_final)
            if inc_avg is not None:
                dict['Inc-Average'] = 100.*float(inc_avg)
        except Exception:
            pass
    # 遗忘 OOD 指标写出（仅在非增量模式）
    if has_forget and (not getattr(args, 'incremental', False)):
        try:
            if forget_fpr is not None:
                dict['forget-FPR'] = 100.*float(forget_fpr)
            if forget_auroc is not None:
                dict['forget-AUROC'] = 100.*float(forget_auroc)
        except Exception:
            pass

    # append retain accuracy if provided by evaluator
    if retain_acc is not None:
        try:
            dict['Retain-Acc'] = 100.*float(retain_acc)
        except Exception:
            dict['Retain-Acc'] = float(retain_acc)


    if not os.path.exists(save_path):
        with open(save_path, 'w') as f:
            writer = DictWriter(f, fieldnames=field_names)
            writer.writeheader()

    # Open CSV file in append mode
    # Create a file object for this file
    with open(save_path, 'a') as f_object:
        dictwriter_object = DictWriter(f_object, fieldnames=field_names)
        dictwriter_object.writerow(dict)
        f_object.close()