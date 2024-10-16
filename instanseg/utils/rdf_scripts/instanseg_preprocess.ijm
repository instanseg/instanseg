/**
 * Simple preprocessing script to apply min/max normalization per channel.
 * 
 * Note that percentile normalization (0.1% and 99.9%) is used by InstanSeg 
 * elsewhere, and *may* give better results (at least if there are outliers).
 * 
 * This ImageJ macro by Pete Bankhead is marked with CC0 1.0.
 * Note that InstanSeg and pre-trained models are under their own licenses.
 */

// Ensure image is 32-bit
run("32-bit");

// Normalize channels separately - we assume 2D
getDimensions(width, height, channels, slices, frames)
Stack.getPosition(channel, slice, frame);
for (c = 1; c <= channels; c++) {
	Stack.setPosition(c, slice, frame);
	getStatistics(area, mean, min, max, std, histogram);
	run("Subtract...", "value=" + min);
	run("Divide...", "value=" + (max - min));
	setMinAndMax(0, 1);
}