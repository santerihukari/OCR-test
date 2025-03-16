# OCR-test

This project was implemented in less than an hour to detect Regions of Interest (RoIs) — potential characters of a certain size — from an image. It includes manual labeling of a subset of these regions, which is then used to detect the remaining numbers.

*Disclaimer: This was done to procrastinate from another project. 🫠*

## Results
- All numbers in the test were correctly predicted. Test image not provided due to security reasons.
- Some non-number characters were incorrectly predicted as numbers.

## Improvements
If I were to implement this again, I would:
1. Apply a projective transformation to the sheet of paper to correct for **scale**, **shear**, **rotation**, and **perspective distortion**. This would normalize the characters and improve detection accuracy.
2. Use the same approach for detection and prediction.
