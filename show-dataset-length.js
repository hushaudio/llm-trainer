const fs = require('fs');
const path = require('path'); // Import path module to handle file paths

const dataRoot = "./data";
let files = [];

// Check if dataRoot is a directory, then filter for .json files only
if (fs.lstatSync(dataRoot).isDirectory()) {
  files = fs.readdirSync(dataRoot)
             .filter(file => path.extname(file) === '.json'); // Only include .json files
} else {
  // If dataRoot is not a directory (i.e., it's a file), check if it's a .json file
  if (path.extname(dataRoot) === '.json') {
    files = [dataRoot];
  }
}

console.log(`Reading files from ${dataRoot}: ${files.length}`);

// Combine all json files, which all contain a single array of objects
const data = files.reduce((acc, file) => {
  try {
    const filep = `${dataRoot}/${file}`;
    const json = JSON.parse(fs.readFileSync(filep, 'utf8'));
    return [...acc, ...json];
  } catch (e) {
    throw new Error(`Error while reading file ${file}: ${e.message}`);
  }
}, []);

console.log(`Combined data length: ${data.length}`);
