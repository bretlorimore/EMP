from jinja2 import Environment, Template
import time
import os
import sys
from ast import literal_eval
import math

import util

OUTPUT_FILE = "output.html"
TRUTH_FILE = "image_data.txt"

HTML = """
<script defer src="https://code.getmdl.io/1.2.1/material.min.js"></script>

<link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Roboto:regular,bold,italic,thin,light,bolditalic,black,medium&amp;lang=en">
<link rel="stylesheet" href="https://fonts.googleapis.com/icon?family=Material+Icons">
<link rel="stylesheet" href="https://code.getmdl.io/1.3.0/material.indigo-pink.min.css">

<!DOCTYPE html>
<html>
<head>
    <meta name="viewport" content="initial-scale=1.0, user-scalable=no">
	<meta charset="utf-8">

    <title>{{ title }}</title>

    <style>
		.content-list {
			width: calc(100% - 50px);
		}
		.mdl-layout__drawer-button .material-icons {
			margin-top: 12px;
		}
		.mdl-data-table {
			margin-left: 10px;
			width: calc(100% - 200px);
		}
		img {
			max-width: 300px;
			max-height: 300px;
		}
	</style>

<script>

    var prev_col_name = " ";
    var ascending = true;

    function hide_all_arrows() {

        document.getElementById("sun_diff_up").style.display = "none";
        document.getElementById("sun_diff_down").style.display = "none";

        document.getElementById("moon_diff_up").style.display = "none";
        document.getElementById("moon_diff_down").style.display = "none";

        document.getElementById("times_up").style.display = "none";
        document.getElementById("times_down").style.display = "none";

    }

    function sun_diff_comparator(row1, row2) {
        var col_val = 2;

        var row1_cell = row1.getElementsByTagName("td")[col_val];
        var row2_cell = row2.getElementsByTagName("td")[col_val];

        var row1_result = row1_cell.innerHTML.split("<br>");
        var row2_result = row2_cell.innerHTML.split("<br>");

        var row1_val = row1_result[1];
        row1_val = parseFloat(row1_val) + Math.abs(parseFloat(row1_result[4]));

        var row2_val = row2_result[1];
        row2_val = parseFloat(row2_val) + Math.abs(parseFloat(row2_result[4]));

        if (row1_val < row2_val) {
            return -1;
        }
        if (row1_val > row2_val) {
            return 1;
        }
        return 0;
    }

    function moon_diff_comparator(row1, row2) {
        var col_val = 3;

        var row1_cell = row1.getElementsByTagName("td")[col_val];
        var row2_cell = row2.getElementsByTagName("td")[col_val];

        var row1_result = row1_cell.innerHTML.split("<br>");
        var row2_result = row2_cell.innerHTML.split("<br>");

        var row1_val = row1_result[1];
        if (row1_val.toLowerCase().includes("no moon")) {
            row1_val = 10000;
        } else {
            row1_val = parseFloat(row1_val) + Math.abs(parseFloat(row1_result[4]));
        }

        var row2_val = row2_result[1];
        if (row2_val.toLowerCase().includes("no moon")) {
            row2_val = 10000;
        } else {
            row2_val = parseFloat(row2_val) + Math.abs(parseFloat(row2_result[4]));
        }

        if (row1_val < row2_val) {
            return -1;
        }
        if (row1_val > row2_val) {
            return 1;
        }
        return 0;
    }

    function times_comparator(row1, row2) {
        var col_val = 4;

        var row1_cell = row1.getElementsByTagName("td")[col_val];
        var row2_cell = row2.getElementsByTagName("td")[col_val];

        var row1_list = row1_cell.getElementsByTagName("ul")[0];
        var row2_list = row2_cell.getElementsByTagName("ul")[0];

        var row1_val = 0.0;
        for (element in row1_list.children) {
            if (typeof(row1_list.children[element].innerHTML) != 'undefined') {
                row1_val += parseFloat(row1_list.children[element].innerHTML.split(":")[1]);
            }
        }

        var row2_val = 0.0;
        for (element in row2_list.children) {
            if (typeof(row2_list.children[element].innerHTML) != 'undefined') {
                row2_val += parseFloat(row2_list.children[element].innerHTML.split(":")[1]);
            }
        }

        if (row1_val < row2_val) {
            return -1;
        }
        if (row1_val > row2_val) {
            return 1;
        }
        return 0;
    }

    function sort_table(n) {

        var table = document.getElementById("eclipse_data_table");
        var table_body = document.getElementById("eclipse_data_table_body");

        var row_array = Array.prototype.slice.call(table_body.children);

        var start = performance.now();

        switch(n) {
            case 2:
                var col_name = "sun_diff";
                row_array.sort(sun_diff_comparator);
                break;
            case 3:
                var col_name = "moon_diff";
                row_array.sort(moon_diff_comparator);
                break;
            case 4:
                var col_name = "times";
                row_array.sort(times_comparator);
                break;
            default:
                console.log("error this isn't possible");
        }

        if(prev_col_name == col_name) {
            ascending = !ascending;
        } else {
            ascending = true;
        }

        if (ascending) {
            for (var i = 0; i < row_array.length; i++) {
                table.children[1].appendChild(row_array[i]);
            }
            var arrow_direction = "up";
        } else {
            for (var i = row_array.length -1; i > -1; i--) {
                table.children[1].appendChild(row_array[i]);
            }
            var arrow_direction = "down";
        }

        hide_all_arrows();
        document.getElementById(col_name + "_" + arrow_direction).style.display = "";

        var end = performance.now();
        var time = end - start;
        console.log('Execution time: ' + time);

        prev_col_name = col_name;
    }

</script>
</head>
<body>

    <div class="mdl-layout mdl-js-layout mdl-layout--fixed-header">
		<header class="mdl-layout__header">
			<div class="mdl-layout__header-row">
				<!-- Title -->
				<span class="mdl-layout-title">{{ title }}</span>
				<!-- Spacer to align links on the right -->
				<div class="mdl-layout-spacer"></div>

			</div>
		</header>
		
		<main class="drawer-tab mdl-layout__content">
			<div class="page-content">

                <h3 id="revision">Git Revision</h3>
                    <p>{{ gitrev }}</p>

		<h3> Test Timestamp:</h3>
			<p>{{ date }}</p>

                <h3>Output Table</h3>

                <table id="eclipse_data_table" class="mdl-data-table mdl-js-data-table">
                    <thead>
                        <tr>
	                        <th class="mdl-data-table__cell--non-numeric">
		                        Original
	                        </th>
	                        <th class="mdl-data-table__cell--non-numeric">
		                        Processed
	                        </th>

	                        <th class="mdl-data-table__cell--non-numeric" onclick="sort_table(2)" style="cursor: pointer;">
		                        Sun Diff (px)
                                <i id="sun_diff_down" style="position: absolute; display: none;" class="material-icons">keyboard_arrow_down</i>
                                <i id="sun_diff_up" style="position: absolute; display: none;" class="material-icons">keyboard_arrow_up</i>
	                        </th>
	                        <th class="mdl-data-table__cell--non-numeric" onclick="sort_table(3)" style="cursor: pointer;">
		                        Moon Diff (px)
                                <i id="moon_diff_down" style="position: absolute; display: none;" class="material-icons">keyboard_arrow_down</i>
                                <i id="moon_diff_up" style="position: absolute; display: none;" class="material-icons">keyboard_arrow_up</i>
	                        </th>
	                        <th class="mdl-data-table__cell--non-numeric" onclick="sort_table(4)" style="cursor: pointer;">
		                        Running times (secs)
                                <i id="times_down" style="position: absolute; display: none;" class="material-icons">keyboard_arrow_down</i>
                                <i id="times_up" style="position: absolute; display: none;" class="material-icons">keyboard_arrow_up</i>
	                        </th>
		                    <th class="mdl-data-table__cell--non-numeric">
		                        Comments
	                        </th>
                        </tr>
                    </thead>
				    <tbody id="eclipse_data_table_body">
                    {% for item in items %}
					    <tr>
						    <td class="mdl-data-table__cell--non-numeric img-cell">
							    <img src="{{item.original}}"> <br>
						    </td>
						    <td class="mdl-data-table__cell--non-numeric">
							    <img src="{{item.processed}}">
						    </td>
						    <td class="mdl-data-table__cell--non-numeric">
							    Center offset: <br>{{item.sun_center_diff}}<br>
                                <br>
                                Radius difference: <br>{{item.sun_rad_diff}}<br>
						    </td>
						    <td class="mdl-data-table__cell--non-numeric">
                                {{ item.no_moon }}
							    Center offset: <br>{{item.moon_center_diff}}<br>
                                <br>
                                Radius difference: <br>{{item.moon_rad_diff}}<br>
						    </td>
						    <td class="mdl-data-table__cell--non-numeric">
						        <ul>
						        {% for time in item.times %}
						            <li> {{ time }} </li>
						        {% endfor %}
						        </ul>
						    </td>
						    <td class="mdl-data-table__cell--non-numeric">
							    <ul>
							    {% for comment in item.comments %}
							        <li> {{ comment }} </li>
							    {% endfor %}
							    </ul>
						    </td>
					    </tr>
                    {% endfor %}
				    </tbody>
			    </table>

			</div>
		</main>
	</div>

</body>
</html>
"""


def calc_position_diff(result, truth):
    result_pos = (result[0], result[1])
    truth_pos = (truth[0], truth[1])

    radius_diff = result[2] - truth[2]

    center_offset = euclidean_distance(result_pos, truth_pos)

    return center_offset, radius_diff


def euclidean_distance(circle1, circle2):
	dx = circle1[0] - circle2[0]
	dy = circle1[1] - circle2[1]
	return math.sqrt((dx ** 2) + (dy ** 2))


def read_metadata(original_path, processed_path, original_bucket, processed_bucket, converter):

    truth_file = open(os.path.join(original_path, TRUTH_FILE), 'r')

    truth_positions = {}
    for line in truth_file:
        tokens = line.split('|')

        position = dict(sun = literal_eval(tokens[2]), moon = literal_eval(tokens[3]))

        truth_positions[os.path.join(processed_path, tokens[0])] = position
    
    f = open(os.path.join(processed_path, "metadata.txt"), 'r')

    metadata_items = []

    for line in f.readlines():
        tokens = line.split('|')
        
        path_tokens = tokens[0].split('/')
        img_name = path_tokens[len(path_tokens) - 1]

        item = dict(
            image_name = img_name, 
            original   = util.gcs_url(img_name, original_bucket),
            processed  = util.gcs_url(converter.get_run_specific_filename(img_name), processed_bucket)
        )
                    
        times = []
        comments = []
        token_count = 0
        for token in tokens:
            if token.startswith('t'):
                tup = literal_eval(token[1:])
                times.append(tup[0] + ":\t" + str(tup[1]))
            elif token.startswith('c'):
                None
            elif token_count > 1 and token != "\n":
                comments.append(token)
            else:
                None
            token_count += 1
                
        item['times'] = times
        item['comments'] = comments

        if truth_positions[tokens[0]]['moon'] is not None:
            moon_center_offset, moon_radius_diff = calc_position_diff(literal_eval(tokens[2][1:]), truth_positions[tokens[0]]['moon'])
            item['moon_center_diff'] = moon_center_offset
            item['moon_rad_diff'] = moon_radius_diff
        else:
            item['moon_center_diff'] = "No Moon in ground truth"
            item['moon_rad_diff'] = "No Moon in ground truth"

        sun_center_offset, sun_radius_diff = calc_position_diff(literal_eval(tokens[1][1:]), truth_positions[tokens[0]]['sun'])

        item['sun_center_diff'] = sun_center_offset
        item['sun_rad_diff'] = sun_radius_diff

        if tokens[5] == "1":
            item['comments'] = '<br>'.join(item.strip() for item in tokens[6].split(';'))

        metadata_items.append(item)

    return metadata_items

    
def build_html_doc(original_path, processed_path, original_bucket, processed_bucket, converter):

    # set date/time for title
    date_time = time.strftime("%c", converter.datetime.timetuple())

    page_title = "Eclipse Image Processor Output"

    metadata = read_metadata(original_path, processed_path, original_bucket, processed_bucket, converter)

    html_path = os.path.join(processed_path, OUTPUT_FILE)
    f = open(html_path, 'w')
    f.write(Environment().from_string(HTML).render(title=page_title, gitrev=converter.git_hash, date=date_time, items=metadata))
    
    return html_path


def main():

    if len(sys.argv) < 5:
        params = 'path/to/original/images path/to/processed/images original_image_gcs_bucket processed_gcs_bucket'
        print('Please run this script in the form:')
        print('\n\t$ python3 {} {}\n'.format(os.path.basename(__file__), params))
        return 

    original_dir, processed_dir, original_bucket, processed_bucket = sys.argv[1:5]

    converter = util.FilenameConverter()    
    html_path = build_html_doc(original_dir, processed_dir, original_bucket, processed_bucket, converter)
    html_path = converter.get_run_specific_filename(html_path)

    converter.commit(processed_dir)

    print('WEB_URL:{}'.format(util.gcs_url(html_path, processed_bucket)))


if __name__ == '__main__':
    main()

