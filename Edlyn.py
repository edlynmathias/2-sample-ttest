import streamlit as st
import numpy as np
from scipy.stats import t

# ------------------------------
# Your Function
# ------------------------------
def t_test_independent_pooled(a, b, alpha=0.05, alternative="two-sided"):
    a = np.array(a)
    b = np.array(b)

    n1, n2 = len(a), len(b)
    xbar1, xbar2 = np.mean(a), np.mean(b)
    s1, s2 = np.std(a, ddof=1), np.std(b, ddof=1)

    sp2 = ((n1-1)*s1**2 + (n2-1)*s2**2) / (n1 + n2 - 2)
    se = np.sqrt(sp2 * (1/n1 + 1/n2))
 
    t_cal = (xbar1 - xbar2) / se
    df = n1 + n2 - 2

    if alternative == "two-sided":
        t_crit = t.ppf(1 - alpha/2, df)
        p_value = 2 * (1 - t.cdf(abs(t_cal), df))
        reject = abs(t_cal) > t_crit

    elif alternative == "greater":
        t_crit = t.ppf(1 - alpha, df)
        p_value = 1 - t.cdf(t_cal, df)
        reject = t_cal > t_crit

    elif alternative == "less":
        t_crit = t.ppf(alpha, df)
        p_value = t.cdf(t_cal, df)
        reject = t_cal < t_crit

    return t_cal, df, p_value, reject


# ------------------------------
# Streamlit UI
# ------------------------------

st.title("Independent Two Sample t-Test (Pooled)")

st.write("Enter sample values separated by commas")

sample1 = st.text_input("Sample 1 (a)", "10,12,14,15,13")
sample2 = st.text_input("Sample 2 (b)", "8,9,11,10,12")

alpha = st.number_input("Significance Level (alpha)", 
                        min_value=0.001, 
                        max_value=0.5, 
                        value=0.05)

alternative = st.selectbox("Alternative Hypothesis",
                           ["two-sided", "greater", "less"])

if st.button("Run Test"):
    try:
        a = [float(x.strip()) for x in sample1.split(",")]
        b = [float(x.strip()) for x in sample2.split(",")]

        t_cal, df, p_value, reject = t_test_independent_pooled(a, b, alpha, alternative)

        st.subheader("Results")
        st.write(f"t statistic: {t_cal:.4f}")
        st.write(f"Degrees of Freedom: {df}")
        st.write(f"p-value: {p_value:.6f}")

        if reject:
            st.success("Decision: Reject Null Hypothesis")
        else:
            st.info("Decision: Fail to Reject Null Hypothesis")

    except:
        st.error("Please enter valid numeric inputs.")