from general128_v09r33 import General128V09R33

CASES = [
("An interior design firm offers installation for $129.00. It includes hanging 4 mirrors, 2 shelves, 1 chandelier, and 10 pictures. They will install additional items for an extra $15.00 per item. Angela has 6 mirrors and 2 chandeliers and 20 pictures that she needs installed/hung. How much will this cost her?", 324),
("I have 10 liters of orange drink that are two-thirds water and I wish to add it to 15 liters of pineapple drink that is three-fifths water. But as I pour it, I spill one liter of the orange drink. How much water is in the remaining 24 liters?", 15),
("Jim spends 2 hours watching TV and then decides to go to bed and reads for half as long. He does this 3 times a week. How many hours does he spend on TV and reading in 4 weeks?", 36),
("Molly is catering a birthday party for her sister and invited 16 people. 10 people want the chicken salad which is $6.50 per person and 6 people want the pasta salad at $6 per person. What is the total amount Molly will pay for the catering?", 101),
("Milo is making a mosaic with chips of glass. It takes twelve glass chips to make every square inch of the mosaic. A bag of glass chips holds 72 chips. Milo wants his mosaic to be three inches tall. If he has two bags of glass chips, how many inches long can he make his mosaic?", 4),
]

g=General128V09R33('http://127.0.0.1:9')
passed=0
for i,(q,gold) in enumerate(CASES):
    ops0=g.math.ops
    r=g.answer_math(q, baseline_hint='0')
    pred=float(r.answer) if r.answer is not None else None
    ok=pred is not None and abs(pred-gold)<1e-9
    passed += int(ok)
    print(i,'gold',gold,'pred',pred,'ok',ok,'ops',g.math.ops-ops0,'trace',r.trace)
print('PASSED',passed,'/',len(CASES))
assert passed == len(CASES)
assert g.math.ops > 0
