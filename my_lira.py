

ef lira_train_models(PATH, device, train_set, test_set, target_model, batch_size, use_DP, noise, norm, delta, dataset_name, n_shadows, pkeep=0.5):
    # def run():
    
    
    for shadow_id in range(n_shadows):
        print(f"Training shadow {shadow_id}")
        # exit()

        seed = np.random.randint(0, 1000000000)
        seed ^= int(time.time())
        pl.seed_everything(seed)

        debug = True
        wandb.init(project="lira", mode="disabled" if debug else "online")
        # wandb.config.update(args)

        # train_ds = train_set
        # test_ds = test_set

        size = len(train_set)
        np.random.seed(seed)
        if n_shadows is not None:
            np.random.seed(0)
            keep = np.random.uniform(0, 1, size=(n_shadows, size))
            order = keep.argsort(0)
            keep = order < int(pkeep * n_shadows)
            keep = np.array(keep[shadow_id], dtype=bool)
            keep = keep.nonzero()[0]
        else:
            keep = np.random.choice(size, size=int(pkeep * size), replace=False)
            keep.sort()
        keep_bool = np.full((size), False)
        keep_bool[keep] = True

        train_ds = torch.utils.data.Subset(train_set, keep)
        # train_dl = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=1)
        # test_dl = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=1)
        # exit()
        train_lira_func(PATH, device, train_ds, test_set, target_model, batch_size, use_DP, noise, norm, delta, dataset_name, shadow_id, keep_bool)
        
from tqdm import tqdm

@torch.no_grad()
def lira_inference( device, train_set, test_set, target_model, batch_size, dataset_name, n_queries=3):


    print("Training model: train set shape", len(train_set), " test set shape: ", len(test_set), ", device: ", device)
    print(f"dataset Name: {dataset_name}")
    # train_dl = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=1)
    # test_dl = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=1)
    train_dl = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False)
    
    savedir = f"exp/{dataset_name}"

    
    
    # Infer the logits with multiple queries
    for path in os.listdir(savedir):
        print(os.path.join(savedir, path, "model.pt"))
        m = target_model
        m.load_state_dict(torch.load(os.path.join(savedir, path, "model.pt")))
        m.to(device)
        m.eval()

        logits_n = []
        for i in range(n_queries):
            logits = []
            for x, _ in tqdm(train_dl):
                x = x.to(device)
                outputs = m(x)
                logits.append(outputs.cpu().numpy())
            logits_n.append(np.concatenate(logits))
        logits_n = np.stack(logits_n, axis=1)
        print(logits_n.shape)
        # Print a few samples from the first query
        # print("First query samples (logits):")
        # for i, sample in enumerate(logits_n[0][:5]):
        #     print(f"Sample {i + 1}: {sample}")
        # exit()

        np.save(os.path.join(savedir, path, "logits.npy"), logits_n)
        # exit()



        
@torch.no_grad()
def lira_inference( device, train_set, test_set, target_model, batch_size, dataset_name, n_queries=3):


    print("Training model: train set shape", len(train_set), " test set shape: ", len(test_set), ", device: ", device)
    print(f"dataset Name: {dataset_name}")
    # train_dl = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=1)
    # test_dl = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=1)
    train_dl = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False)
    
    savedir = f"exp/{dataset_name}"

    
    
    # Infer the logits with multiple queries
    for path in os.listdir(savedir):
        print(os.path.join(savedir, path, "model.pt"))
        m = target_model
        m.load_state_dict(torch.load(os.path.join(savedir, path, "model.pt")))
        m.to(device)
        m.eval()

        logits_n = []
        for i in range(n_queries):
            logits = []
            for x, _ in tqdm(train_dl):
                x = x.to(device)
                outputs = m(x)
                logits.append(outputs.cpu().numpy())
            logits_n.append(np.concatenate(logits))
        logits_n = np.stack(logits_n, axis=1)
        print(logits_n.shape)
        # Print a few samples from the first query
        # print("First query samples (logits):")
        # for i, sample in enumerate(logits_n[0][:5]):
        #     print(f"Sample {i + 1}: {sample}")
        # exit()

        np.save(os.path.join(savedir, path, "logits.npy"), logits_n)
        # exit()


class Lira_score_process:
    def __init__(self, ATTACK_PATH, attack_name, train_ds, dataset_name, ntest):
        self.train_ds = train_ds
        self.savedir = f"exp/{dataset_name}"
        self.ntest = ntest # number of test samples
        
        self.all_labels = [label for (_, label) in train_ds]
        
        self.scores = []
        self.keep = []

        self.fpr_tpr_file_path = ATTACK_PATH + "_FPR_TPR_" + attack_name + "_.csv"

    def load_one(self, path):
        """
        This loads logits and converts them to scored predictions.
        """
        opredictions = np.load(os.path.join(path, "logits.npy"))  # [n_examples, n_augs, n_classes]
        print(os.path.join(path, "logits.npy"))
        # exit()
        # Be exceptionally careful.
        # Numerically stable everything, as described in the paper.
        predictions = opredictions - np.max(opredictions, axis=-1, keepdims=True)
        predictions = np.array(np.exp(predictions), dtype=np.float64)
        predictions = predictions / np.sum(predictions, axis=-1, keepdims=True)
        
        
        labels = self.all_labels

        COUNT = predictions.shape[0]
        y_true = predictions[np.arange(COUNT), :, labels[:COUNT]]

        print("mean acc", np.mean(predictions[:, 0, :].argmax(1) == labels[:COUNT]))

        predictions[np.arange(COUNT), :, labels[:COUNT]] = 0
        y_wrong = np.sum(predictions, axis=-1)

        logit = np.log(y_true + 1e-45) - np.log(y_wrong + 1e-45)
        np.save(os.path.join(path, "scores.npy"), logit)

        

    def load_stats(self):
        with mp.Pool(8) as p:
            p.map(self.load_one, [os.path.join(self.savedir, x) for x in os.listdir(self.savedir)])

    def lira_score(self):
        """
        Process logits and compute scores for the given dataset.
        """
        # savedir = f"exp/{dataset_name}"
        self.load_stats()


    def load_data(self):
        """
        Load our saved scores and then put them into a big matrix.
        """
        # global scores, keep
        scores = []
        keep = []

        for path in os.listdir(self.savedir):
            scores.append(np.load(os.path.join(self.savedir, path, "scores.npy")))
            keep.append(np.load(os.path.join(self.savedir, path, "keep.npy")))
        
        self.scores = np.array(scores)
        self.keep = np.array(keep)
        # print("Type of keep:", type(keep))
        # print("Type of keep:", type(self.keep))
        # print("Shape of keep:", np.array(self.keep).shape)
        # print("Size of keep:", len(self.keep))
        # Print a few values of keep for debugging
        # print("Few values of keep (first 5):", self.keep[:5])
        # exit()
        # Print a few of the scores and exit
        # print("load_data- Few scores (first 5):", scores[:5])
        # exit()
        return self.scores, self.keep
    
    def sweep(score, x):
        """
        Compute a ROC curve and then return the FPR, TPR, AUC, and ACC.
        """
        fpr, tpr, _ = roc_curve(x, -score)
        acc = np.max(1 - (fpr + (1 - tpr)) / 2)
        return fpr, tpr, auc(fpr, tpr), acc
    
    def generate_ours_offline(self,in_size=100000, out_size=100000, fix_variance=False ):
        """
        Fit a single predictive model using keep and scores in order to predict
        if the examples in check_scores were training data or not, using the
        ground truth answer from check_keep.
        """
        ntest = self.ntest
        
        keep = self.keep[:-ntest]      # Training keep: all but the last ntest elements.
        scores = self.scores[:-ntest]  # Training scores: all but the last ntest elements.
        check_keep = self.keep[-ntest:]       # Test keep: the last ntest elements.
        check_scores = self.scores[-ntest:]   # Test scores: the last ntest elements.


        dat_in = []
        dat_out = []

        for j in range(scores.shape[1]):
            dat_in.append(scores[keep[:, j], j, :])
            dat_out.append(scores[~keep[:, j], j, :])

        out_size = min(min(map(len, dat_out)), out_size)

        dat_out = np.array([x[:out_size] for x in dat_out])

        mean_out = np.median(dat_out, 1)

        if fix_variance:
            std_out = np.std(dat_out)
        else:
            std_out = np.std(dat_out, 1)

        prediction = []
        answers = []
        for ans, sc in zip(check_keep, check_scores):
            score = scipy.stats.norm.logpdf(sc, mean_out, std_out + 1e-30)

            prediction.extend(score.mean(1))
            answers.extend(ans)
        return prediction, answers

    def generate_ours(self,in_size=100000, out_size=100000, fix_variance=False):
        """
        Fit a two predictive models using keep and scores in order to predict
        if the examples in check_scores were training data or not, using the
        ground truth answer from check_keep.
        """
        # keep, scores, check_keep, check_scores = self.keep[:-ntest], self.scores[:-ntest], self.keep[-ntest:], self.scores[-ntest:]
        
        # prediction, answers = fn(self.keep[:-ntest], self,scores[:-ntest], self.keep[-ntest:], self.scores[-ntest:])

        ntest = self.ntest

        keep = self.keep[:-ntest]      # Training keep: all but the last ntest elements.
        scores = self.scores[:-ntest]  # Training scores: all but the last ntest elements.
        check_keep = self.keep[-ntest:]       # Test keep: the last ntest elements.
        check_scores = self.scores[-ntest:]   # Test scores: the last ntest elements.


        print("generate_ours-Type of keep:", type(keep))
        print("Shape of keep:", np.array(keep).shape)
        print("Size of keep:", len(keep))
        # exit()
        

        dat_in = []
        dat_out = []

        # # Print shapes and a few values of scores
        # print("Shape of scores:", np.array(scores).shape)
        # print("Shape of keep:", np.array(keep).shape)
        # print("Few values of scores:", scores[:2])  # Print first two entries for brevity
        # print("Few values of keep:", keep[:2])  # Print first two entries for brevity
        # exit()

        for j in range(scores.shape[1]):
            dat_in.append(scores[keep[:, j], j, :])
            dat_out.append(scores[~keep[:, j], j, :])

        in_size = min(min(map(len, dat_in)), in_size)
        out_size = min(min(map(len, dat_out)), out_size)

        dat_in = np.array([x[:in_size] for x in dat_in])
        dat_out = np.array([x[:out_size] for x in dat_out])

        mean_in = np.median(dat_in, 1)
        mean_out = np.median(dat_out, 1)

        if fix_variance:
            std_in = np.std(dat_in)
            std_out = np.std(dat_in)
        else:
            std_in = np.std(dat_in, 1)
            std_out = np.std(dat_out, 1)

        prediction = []
        answers = []
        for ans, sc in zip(check_keep, check_scores):
            pr_in = -scipy.stats.norm.logpdf(sc, mean_in, std_in + 1e-30)
            pr_out = -scipy.stats.norm.logpdf(sc, mean_out, std_out + 1e-30)
            score = pr_in - pr_out

            prediction.extend(score.mean(1))
            answers.extend(ans)

        return prediction, answers

    def do_plot(self, fn, keep, scores, ntest, legend="", metric="auc", sweep_fn=sweep, **plot_kwargs):
        """
        Generate the ROC curves by using ntest models as test models and the rest to train.
        """
        # Print shapes and a few values of scores
        # print("Shape of scores:", np.array(scores).shape)
        # print("Shape of keep:", np.array(keep).shape)
        # keep = self.keep
        # scores = self.scores
        # print("Type of keep:", type(keep))
        # print("Shape of keep:", np.array(keep).shape)
        # print("Size of keep:", len(keep))
        # print("Few values of scores:", scores[:2])  # Print first two entries for brevity
        # print("Few values of keep:", keep)  # Print first two entries for brevity
        # exit()

        prediction, answers = fn()
        # print("Predictions (first 5):", prediction[:5])
        # print("Answers (first 5):", answers[:5])
        # exit()
        fpr, tpr, auc, acc = sweep_fn(np.array(prediction), np.array(answers, dtype=bool))

       
         # Save fpr and tpr to a CSV file
        df_fpr_tpr = pd.DataFrame({'FPR': fpr, 'TPR': tpr})
        df_fpr_tpr.to_csv(self.fpr_tpr_file_path, index=False)
        print(f"m_lira_saved ROC curve info")
    
        low = tpr[np.where(fpr < 0.001)[0][-1]]

        print("Attack %s   AUC %.4f, Accuracy %.4f, TPR@0.1%%FPR of %.4f" % (legend, auc, acc, low))

        metric_text = ""
        if metric == "auc":
            metric_text = "auc=%.3f" % auc
        elif metric == "acc":
            metric_text = "acc=%.3f" % acc

        # plt.plot(fpr, tpr, label=legend + metric_text, **plot_kwargs)
        return (acc, auc)

    def fig_fpr_tpr(self):

        self.load_data()
        # exit()
        plt.figure(figsize=(4, 3))

        # self.do_plot(self.generate_ours, self.keep, self.scores, 1, "Ours (online)\n", metric="auc")
        # self.do_plot(self.generate_ours, self.keep, self.scores, 1, "Ours (online)\n", "auc")

        self.do_plot(functools.partial(self.generate_ours, fix_variance=True), self.keep, self.scores, 1, "Ours (online, fixed variance)\n", metric="auc")


def create_subsets(train_dataset, N):
    """
    Randomly create N subsets from train_dataset so that
    each sample appears in exactly N/2 of them.

    Args:
        train_dataset: A list (or similar) of training samples, e.g. [(x1, y1), (x2, y2), ...]
        N (int): Number of subsets/models to create. Must be even.
    Returns:
        subsets (list of lists): subsets[i] is a list of samples for model i.
    """
    assert N % 2 == 0, "N must be even so each sample can appear in N/2 subsets."
    M = len(train_dataset)

    # Create empty "bins" for each subset
    subsets = [[] for _ in range(N)]

    for i in range(M):
        # Randomly pick which N/2 subsets this sample belongs to
        chosen = random.sample(range(N), k=N//2)
        for subset_idx in chosen:
            subsets[subset_idx].append(train_dataset[i])

    return subsets



def main():
    if attack_name == "m_lira": 
                print("LiRA inference.")
                
                if args.lira_train:
                    lira_train_models(MODEL_SAVE_PATH, device, target_train, target_test, target_model, batch_size, use_DP, noise, norm, delta, dataset_name, n_shadows=16, pkeep=0.5)
                
                if args.lira_inference:
                    lira_inference(device, target_train, target_test, target_model, batch_size, dataset_name, n_queries=args.aug)

                if args.lira_roc:
                        
                    # lira_scores()
                    # lira_score(dataset_name, target_train)
                    # print("Type of target_train:", type(target_train))
                    lira_process = Lira_score_process(MODEL_SAVE_PATH, attack_name, target_train, dataset_name, ntest=1)
                    lira_process.lira_score()
                    lira_process.fig_fpr_tpr()
                