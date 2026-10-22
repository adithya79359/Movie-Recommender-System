### Computed conclusion

- Validation selected **Popularity** using NDCG@10.
- Its test NDCG@10 is **0.2329**; the user-bootstrap 95% interval is **[0.2015, 0.2669]**.
- Its test Precision@10 is **0.0899**, Recall@10 is **0.3529**, and coverage is **12.88%**.
- **Ridge** was the best supervised ranker on validation. The lowest observed test rating RMSE belongs to **Ridge**; this is a different objective.
- Test ranking evaluation includes **258** users and **295** reference-observed movies. **11** test ratings refer to movies outside that candidate catalogue.
- **Interaction SVD** has the highest observed test NDCG@10 (**0.2774**). This descriptive test result does not replace the validation-selected method.
- There are **304** candidate movie links, covering **6.67%** of the historical movie catalogue. These results cannot be generalized to all JioHotstar users or movies.

The tables report measured results, not a promised accuracy. Model choice is based on validation rather than selecting whichever model happens to win on the test set.
