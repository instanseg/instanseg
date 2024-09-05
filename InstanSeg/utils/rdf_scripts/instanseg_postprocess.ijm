/*
 * Add ROIs to an overlay, from a multichannel labeled image.
 * 
 * Note that batch mode is *very* important for performance, 
 * although it may be set automatically by DeepImageJ.
 * 
 * This ImageJ macro by Pete Bankhead is marked with CC0 1.0.
 * Note that InstanSeg and pre-trained models are under their own licenses.
 */
 
setBatchMode(true);
getDimensions(width, height, numChannels, numSlices, numFrames)
Stack.getPosition(channel, slice, frame);
Overlay.clear();
for (c = 1; c <= numChannels; c++) {
	Stack.setPosition(c, slice, frame);
	prefix = getNamePrefix(c, numChannels);
	createOverlay(prefix, c);
	resetMinAndMax();
}
Stack.setPosition(channel, slice, frame);
resetMinAndMax();

// Use Glasbey if we have a single channel
if (numChannels == 1)
	run("Glasbey");


/*
 * Get a prefix to use when naming ROIs.
 */
function getNamePrefix(currentChannel, nChannels) {
	if (nChannels == 1)
		return "";
	if (nChannels == 2) {
		if (currentChannel == 1)
			return "Nucleus-";
		else
			return "Cell-";
	}
	return "C" + currentChannel + "-";
}

/*
 * Create an overlay by looping through pixels and using the Wand.
 * It'd be nice to use analyze particles, but we can't because our labels may touch.
 */
function createOverlay(namePrefix, group) {
	width = getWidth();
	height = getHeight();
	id = getImageID();
	getStatistics(area, mean, min, max);
	newImage("Visited", "32-bit black", width, height, 1);
	idVisited = getImageID();
	selectImage(id);
	nextLabel = 1;
	Stack.getPosition(channel, slice, frame);
	for (y = 0; y < height; y++) {
		for (x = 0; x < width; x++) {
			pixelLabel = getPixel(x, y);
			if (pixelLabel > 0) {
				selectImage(idVisited);
				maskValue = getPixel(x, y);
				selectImage(id);
				if (maskValue == 0) {
					doWand(x, y, 0, "8-connected");
					selectImage(idVisited);
					run("Restore Selection");
					setColor(nextLabel);
					fill();
					selectImage(id);
					if (namePrefix != "")
						Roi.setName(namePrefix + pixelLabel);
					else
						Roi.setName(pixelLabel);
					Roi.setGroup(group);
					// Uncomment this if you want to add a position 
					// (usually this is more trouble than it's worth)
					// Roi.setPosition(channel, slice, frame);
					Overlay.addSelection();
					nextLabel++;
				}
			}
		}
	}
	run("Select None");
}
