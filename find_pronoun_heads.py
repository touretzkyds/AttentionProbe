import pandas as pd

# Load attention results
att_df = pd.read_csv("/home/navya/AttentionProbe/attention_results_no_correctness.csv")

# We compute an "importance score":
# for his: frac_man
# for her: frac_woman

scores = []

for head in sorted(att_df["head"].unique()):
    head_data = att_df[att_df["head"] == head]

    # his → man resolution strength
    his_rows = head_data[head_data["sentence"].str.contains("his")]
    his_score = his_rows["frac_man"].mean() if not his_rows.empty else 0    

    # her → woman resolution strength
    her_rows = head_data[head_data["sentence"].str.contains("her")]
    her_score = her_rows["frac_woman"].mean() if not her_rows.empty else 0

    total_score = his_score + her_score

    scores.append({
        "head": head,
        "his_score": his_score,
        "her_score": her_score,
        "total_score": total_score
    })

score_df = pd.DataFrame(scores)
score_df = score_df.sort_values(by="total_score", ascending=False)

print("=== MOST IMPORTANT HEADS FOR PRONOUN RESOLUTION ===")
print(score_df.head(10))

# Save results
score_df.to_csv("important_heads.csv", index=False)
