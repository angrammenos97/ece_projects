function sort_pictures
	if set -q argv[1]; and set -q argv[2]
		if set -l pictures (find $argv[1]/**.jpg)
			mkdir -p $argv[2]
			for p in $pictures
				set pyear (date +%Y -r $p)
				set pmonth (date +%B -r $p)
				set pdate (date +%d -r $p)
				mkdir -p $argv[2]/$pyear/$pmonth/$pdate
				cp $p $argv[2]/$pyear/$pmonth/$pdate/(basename $p)
			end
		end
	else
		echo "No input or output folder given"
	end
end
