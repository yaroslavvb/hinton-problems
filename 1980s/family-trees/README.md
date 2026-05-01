# Family trees / kinship task

**Source:** Hinton (1986), "Learning distributed representations of concepts", CogSci 8.
**Demonstrates:** Backprop discovers semantic features (nationality, generation, family branch) not explicit in the input. Hinton's most cited demonstration of distributed representation learning.

Two isomorphic family trees (English, Italian), 12 people each, 12 relations: father, mother, husband, wife, son, daughter, uncle, aunt, brother, sister, nephew, niece. 104 valid (person1, relation, person2) triples; train on 100, test on 4.

Architecture: 24-input + 12-relation -> two 6-unit encoders -> 12-unit central -> 6-unit decoder -> 24-unit output. The 6-unit person-encoding layer spontaneously develops interpretable units encoding nationality, generation, branch.
