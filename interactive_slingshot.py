import os,imp,argparse,sys
from slingshot_logos import SLINGSHOT
import slingshot_config
CONFIG={} if not slingshot_config.CONFIG else slingshot_config.CONFIG


def interactive(parser, SLING_EXT=['py','R']):
	import readline
	from tab_completer import tabCompleter
	tabber=tabCompleter()
	args=parser.parse_args()
	readline.set_completer_delims('\t')
	if 'libedit' in readline.__doc__:
		readline.parse_and_bind("bind ^I rl_complete")
	else:
		readline.parse_and_bind("tab: complete")

	arg2help=dict([(x.dest,x.help) for x in parser.__dict__['_actions']])

	print SLINGSHOT
	longest_line = max(len(line.rstrip()) for line in SLINGSHOT.split('\n'))
	HR='\n'+'-'*longest_line+'\n'
	#print "### SLINGSHOT v0.1 ###"
	print "## SLINGSHOT v0.1: interactive mode (see \"slingshot --help\" for more)"
	#print parser.format_help()

	try:
		# SLING
		path_slings = CONFIG.get('PATH_SLINGS','')
		SLING_EXT = CONFIG.get('SLING_EXT','')
		if path_slings and os.path.exists(path_slings) and os.path.isdir(path_slings):
			slings=[fn for fn in os.listdir(path_slings) if fn.split('.')[-1] in SLING_EXT]
			sling_str='  '.join(['(%s) %s' % (si+1, sl) for si,sl in enumerate(slings)])
		while not args.sling:
			readline.set_completer(tabber.pathCompleter)
			print '\n>> SLING: '+arg2help['sling']
			if path_slings and slings:
				print '          [numerical shortcuts for slings found in {dir}]\n          {slings}'.format(dir=path_slings, slings=sling_str)
			sling = raw_input('>> ').strip()
			if sling.isdigit() and 0<=int(sling)-1<len(slings):
				args.sling=os.path.join(path_slings,slings[int(sling)-1])
			elif not os.path.exists(sling):
				print "!! filename does not exist"
			elif not sling.split('.')[-1] in SLING_EXT:
				print "!! filename does not end in one of the acceptable file extensions [%s]" % ', '.join(SLING_EXT)
			else:
				args.sling=sling

		# STONE
		print HR
		import imp,inspect
		sling = imp.load_source('sling', args.sling)
		functions = sling.STONES if hasattr(sling,'STONES') and sling.STONES else sorted([x for x,y in inspect.getmembers(sling, inspect.isfunction)])
		functions_str='  '.join(['(%s) %s' % (si+1, sl) for si,sl in enumerate(functions)])
		tabber.createListCompleter(functions)
		while not args.stone:
			readline.set_completer(tabber.listCompleter)
			#prompt='\n>> STONE: {help}\noptions: {opts}\n>>'.format(help=arg2help['stone'], opts=', '.join(functions))
			#prompt='\n>> STONE: {help}\n>>'.format(help=arg2help['stone'])
			print '>> STONE: '+arg2help['stone']
			#print '          [options]: '+functions_str
			print '          '+functions_str
			stone = raw_input('>> ').strip()
			if stone.isdigit() and 0<=int(stone)-1<len(functions):
				args.stone=functions[int(stone)-1]
			elif not stone in functions:
				print "!! function not in file"
			else:
				args.stone=stone

		# PATH
		print HR
		path_pathlists = CONFIG.get('PATH_PATHLISTS','')
		opener='>> PATH: '
		opener_space=' '*len(opener)
		if path_pathlists and os.path.exists(path_pathlists) and os.path.isdir(path_pathlists):
			pathlists=[fn for fn in os.listdir(path_pathlists) if not fn.startswith('.')]
			joiner='\n'+opener_space
			pathlists_str=''.join(['\n%s(%s) %s' % (opener_space,si+1, sl) for si,sl in enumerate(pathlists)])
		while not args.path:
			readline.set_completer(tabber.pathCompleter)
			print opener+arg2help['stone']
			if path_slings and slings:
				print opener_space + '[numerical shortcuts for pathlists found in {dir}]{pathlists}'.format(dir=path_slings, pathlists=pathlists_str)
			path = raw_input('>> ').strip()
			if path.isdigit() and 0<=int(path)-1<len(pathlists):
				args.path=os.path.join(path_pathlists,pathlists[int(path)-1])
			elif not os.path.exists(path):
				print "!! filename or directory does not exist"
			elif os.path.isdir(path):
				args.ext = raw_input('\n>> EXT: '+arg2help['ext']+'\n>> ').strip()
				args.path=path
			else:
				args.path=path


	except (KeyboardInterrupt,EOFError) as e:
		print '\n>> goodbye'
		exit()

	return args
