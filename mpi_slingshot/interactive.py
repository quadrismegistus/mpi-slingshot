from __future__ import absolute_import
from __future__ import print_function
import os,imp,argparse,sys
from .logos import SLINGSHOT
from .config import CONFIG
from .slingshot import is_csv
from six.moves import input


def interactive(parser, SLING_EXT=['py','R']):
	slings=None
	import readline
	from .tab_completer import tabCompleter
	tabber=tabCompleter()
	args=parser.parse_args()
	readline.set_completer_delims('\t')
	if 'libedit' in readline.__doc__:
		readline.parse_and_bind("bind ^I rl_complete")
	else:
		readline.parse_and_bind("tab: complete")

	arg2help=dict([(x.dest,x.help) for x in parser.__dict__['_actions']])

	print(SLINGSHOT)
	longest_line = max(len(line.rstrip()) for line in SLINGSHOT.split('\n'))
	HR='\n'+'-'*longest_line+'\n'
	#print "### SLINGSHOT v0.1 ###"
	print("## SLINGSHOT v0.1: interactive mode (see \"slingshot --help\" for more)")
	#print parser.format_help()

	try:
		# SLING
		path_slings = CONFIG.get('PATH_SLINGS','')
		SLING_EXT = CONFIG.get('SLING_EXT','')
		if path_slings and os.path.exists(path_slings) and os.path.isdir(path_slings):
			slings=sorted([fn for fn in os.listdir(path_slings) if fn.split('.')[-1] in SLING_EXT])
			sling_str='  '.join(['(%s) %s' % (si+1, sl) for si,sl in enumerate(slings)])
		while not args.sling:
			readline.set_completer(tabber.pathCompleter)
			print('\n>> SLING: '+arg2help['sling'])
			if path_slings and slings:
				print('          [numerical shortcuts for slings found in\n          [{dir}]\n          {slings}'.format(dir=path_slings, slings=sling_str))
			sling = input('>> ').strip()
			if sling.isdigit() and 0<=int(sling)-1<len(slings):
				args.sling=os.path.join(path_slings,slings[int(sling)-1])
			elif not os.path.exists(sling):
				print("!! filename does not exist")
			elif not sling.split('.')[-1] in SLING_EXT:
				print("!! filename does not end in one of the acceptable file extensions [%s]" % ', '.join(SLING_EXT))
			else:
				args.sling=sling

		# STONE
		print(HR)
		if args.sling.endswith('.py'):
			import imp,inspect
			sling = imp.load_source('sling', args.sling)
			functions = sling.STONES if hasattr(sling,'STONES') and sling.STONES else sorted([x for x,y in inspect.getmembers(sling, inspect.isfunction)])
			functions_str='  '.join(['(%s) %s' % (si+1, sl) for si,sl in enumerate(functions)])
			tabber.createListCompleter(functions)
		else:
			functions_str=''
		while not args.stone:

			#prompt='\n>> STONE: {help}\noptions: {opts}\n>>'.format(help=arg2help['stone'], opts=', '.join(functions))
			#prompt='\n>> STONE: {help}\n>>'.format(help=arg2help['stone'])
			print('>> STONE: '+arg2help['stone'])
			#print '          [options]: '+functions_str
			if functions_str:
				readline.set_completer(tabber.listCompleter)
				print('          '+functions_str)
			stone = input('>> ').strip()
			if stone.isdigit() and 0<=int(stone)-1<len(functions):
				args.stone=functions[int(stone)-1]
			elif functions_str and not stone in functions:
				print("!! function not in file")
			else:
				args.stone=stone

		# PATH

		try:
			import llp
			import pandas as pd
			print('>> CORPUS: Type the number or name of an LLP corpus')
			num2cname={}
			for ci,(corpus,cdx) in enumerate(sorted(llp.corpus.load_manifest().items())):
				num2cname[ci+1]=corpus
				print('\t({num}) {name} ({desc})'.format(num=str(ci+1).zfill(2), desc=cdx['desc'], name=cdx['name']))
			#pd.options.display.max_colwidth = 100
			#print(pd.DataFrame(llp.corpus.load_manifest()).T[['desc']])
			llp_input = input('>> ').strip()
			if llp_input.strip().isdigit():
				cnum=int(llp_input.strip())
				if cnum in num2cname:
					cname=num2cname[cnum]
					if corpus: args.llp_corpus=cname
			else:
				#corpus=llp.load_corpus(llp_input)
				#if corpus: args.llp_corpus=corpus
				args.llp_corpus=llp_input.strip()

		except ImportError:
			pass

		if not args.llp_corpus:
			print(HR)
			path_pathlists = CONFIG.get('PATH_PATHLISTS','')
			opener='>> PATH: '
			opener_space=' '*len(opener)
			pathlists_str=''
			if path_pathlists and os.path.exists(path_pathlists) and os.path.isdir(path_pathlists):
				pathlists=[fn for fn in os.listdir(path_pathlists) if not fn.startswith('.')]
				joiner='\n'+opener_space
				pathlists_str=''.join(['\n%s(%s) %s' % (opener_space,si+1, sl) for si,sl in enumerate(pathlists)])
			while not args.path:
				readline.set_completer(tabber.pathCompleter)
				#print opener+arg2help['stone']
				print(opener+'Enter a path either to a pathlist text file, or to a directory of texts')
				if path_slings and slings:
					print('{space}[numerical shortcuts for pathlists found in\n{space}[{dir}]{pathlists}'.format(dir=path_slings, pathlists=pathlists_str,space=opener_space))
				path = input('>> ').strip()
				if path.isdigit() and 0<=int(path)-1<len(pathlists):
					args.path=os.path.join(path_pathlists,pathlists[int(path)-1])
				elif not os.path.exists(path):
					print("!! filename or directory does not exist")
				elif os.path.isdir(path):
					args.ext = input('\n>> EXT: '+arg2help['ext']+'\n>> ').strip()
					args.path=path
				elif is_csv(path):
					args.path=path
					args.pathkey=input('\n>> COLUMN: '+arg2help['pathkey']+'\n>> ').strip()
					args.pathprefix=input('\n>> PREFIX: '+arg2help['pathkey']+'\n>> ').strip()
					args.pathsuffix=input('\n>> SUFFIX: '+arg2help['pathkey']+'\n>> ').strip()
				else:
					args.path=path

		print(HR)
		print('OPTIONAL SECTION')
		module='.'.join(os.path.basename(args.sling).split('.')[:-1])
		#default_savedir='/'.join(['results_slingshot',module,args.stone,now()])
		default_savedir='/'.join(['results_slingshot',module,args.stone])

		args.sbatch = input('\n>> SBATCH: Add to the SLURM/Sherlock process queue via sbatch? [N]\n>> (Y/N) ').strip().lower()=='y'
		if args.sbatch:
			args.parallel = input('\n>> PARALLEL: '+arg2help['parallel']+' [4]\n>> ').strip()

			hours = input('\n>> HOURS: '+arg2help['hours']+' [1]\n>> ').strip()
			hours = ''.join([x for x in hours if x.isdigit()])
			args.hours = parser.get_default('hours') if not hours else hours

			mem = input('\n>> MEMORY: '+arg2help['mem']+' [2G]\n>> ').strip()
			args.mem = parser.get_default('mem') if not mem else mem
		else:
			args.debug = input('\n>> DEBUG: %s? [N]\n>> (Y/N) ' % arg2help['debug']).strip().lower()=='y'

		args.nosave = input('\n>> SAVE: Save results? [Y]\n>> (Y/N) ').strip().lower()=='n'
		if not args.nosave:
			args.savedir = input('\n>> SAVEDIR: Directory to store results in [%s]' % default_savedir  + '\n>> ').strip()
			args.cache = input('\n>> CACHE: Cache partial results? [Y]\n>> (Y/N) ').strip().lower()=='y'
			mfw = input('\n>> MFW: %s' % arg2help['mfw'] + '\n>> ').strip()
			args.mfw=mfw if mfw else parser.get_default('mfw')

		args.quiet = input('\n>> QUIET: %s? [N]\n>> (Y/N) ' % arg2help['quiet']).strip().lower()=='y'
		args.limit = input('\n>> LIMIT: '+arg2help['limit']+' [None]\n>> ').strip()

		print()

	except (KeyboardInterrupt,EOFError) as e:
		print('\n>> goodbye')
		exit()

	return args


def now(now=None,seconds=False):
	import datetime as dt
	if not now:
		now=dt.datetime.now()
	elif type(now) in [int,float,str]:
		now=dt.datetime.fromtimestamp(now)

	return '{0}{1}{2}-{3}{4}{5}'.format(now.year,str(now.month).zfill(2),str(now.day).zfill(2),str(now.hour).zfill(2),str(now.minute).zfill(2),'-'+str(now.second).zfill(2) if seconds else '')
