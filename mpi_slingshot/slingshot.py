from __future__ import absolute_import
from __future__ import print_function
import os,sys,codecs,numpy as np,random,imp,time,random
#try:
	#import ujson as json
import simplejson as json
#except ImportError:
#	import json
from datetime import datetime as dt
import unicodecsv as csv
from collections import defaultdict,Counter
from .config import CONFIG
import six
from six.moves import range
DEFAULT_PATH_KEY='_path'
DEFAULT_EXT = 'txt'

PATH_KEY=CONFIG.get('PATH_KEY','')
PATH_EXT=CONFIG.get('PATH_EXT','').replace('.','')

if not PATH_KEY: PATH_KEY=DEFAULT_PATH_KEY
if not PATH_EXT: PATH_EXT=DEFAULT_PATH_EXT

TXT_MAXCOLS=25000



def slingshot_single_shot(stone,path):
	return stone(path)

def slingshot(path_sling=None,stone_name=None,stone_args=None,paths=None,llp_corpus=None,limit=None,path_source=None,stone=None,path_key=PATH_KEY,path_ext=None,path_prefix='',path_suffix='',cache_results=True,cache_path=None,save_results=True,results_dir=None,shuffle_paths=True,stream_results=True,save_txt=True,txt_maxcols=TXT_MAXCOLS,sling_args=[],sling_kwargs={},num_runs=1,oneshot=False,llp_pass_text=False,llp_method='',progress_bar=False):
	"""
	Main function
	"""

	# Load code-sling and stone-function
	if not stone and not llp_method:
		stone=load_stone_in_sling(path_sling,stone_name)
		if not stone:
			return

	# One shot?
	if oneshot:
		return stone(path_source,*sling_args,**sling_kwargs)

	# Load paths
	all_paths = None
	if llp_corpus and llp_corpus!='None':
		try:
			import llp
			corpus = llp.load_corpus(llp_corpus)
			#print(llp_corpus, corpus)
			all_paths = [(text if (llp_pass_text or llp_method) else text.path) for text in corpus.texts()]
			#print(all_paths[:10])
		except ImportError:
			pass

	if not all_paths:
		all_paths = load_paths(path_source,path_ext,limit,shuffle_paths,path_key,path_prefix,path_suffix) if not paths else paths

	# Multiply paths by runs
	all_paths = [(path,run+1) for path in all_paths for run in range(num_runs)]

	# Break if these weren't returned
	if not all_paths:
		#print '!!',[path_sling,stone_name,path_source,path_ext]
		return

	# Save cache dir
	if cache_results and not cache_path:
		if results_dir: cache_path=os.path.join(results_dir,'cache')


	# Start MPI
	from mpi4py import MPI
	t1 = dt.now()
	comm = MPI.COMM_WORLD
	size = comm.Get_size()
	rank = comm.Get_rank()
	print('>> [Slingshot] initializing MPI with size %s and rank %s' % (size,rank))


	# Am I the seed process?
	if rank == 0:
		# Make cache folder
		if cache_results and cache_path and not os.path.exists(cache_path): os.makedirs(cache_path)

		# Farm out paths to other processes
		segments = np.array_split(all_paths,size) if size>1 else [all_paths]
		print('>> [Slingshot]: %s paths divided into %s segments' % (len(all_paths), len(segments)))

	# Or am I a process created by the seed?
	else:
		segments = None

	# Scatter the segments (if rank is 0?)
	segment = comm.scatter(segments, root=0)

	# Parse the segment.
	paths = segment

	# cache results?
	cache_writer=None
	if cache_results and cache_path:
		cache_fn = 'results.rank=%s.jsonl' % str(rank).zfill(4)
		cache_fnfn = os.path.join(cache_path,cache_fn)
		cache_writer=codecs.open(cache_fnfn, mode='w',encoding='utf-8')



	# let's go! loop over the paths
	results=[]
	num_paths=len(paths)
	pronoun='their'
	zlen=len(str(num_paths))
	zlen_rank=len(str(size))

	#progress_bar=True
	if progress_bar:
		from tqdm import tqdm
		looper=tqdm(paths,file=sys.stdout,desc='Slingshot-%s' % str(rank+1).zfill(3),position=rank,ncols=100)#,mininterval=1.0)
	else:
		looper=paths

	for i,(path,run) in enumerate(looper):
		#################################################
		# THIS IS WHERE THE STONE FITS INTO THE SLINGSHOT

		sling_kwargs2=dict(list(sling_kwargs.items()))
		#sling_kwargs2['results_dir']=results_dir
		if num_runs>1: sling_kwargs2['run']=run

		if llp_method:
			text = path
			path = text.addr
			stone = getattr(text,llp_method)
			result = stone(*sling_args,**sling_kwargs2)

		else:
			try:
				result=stone(path,*sling_args,**sling_kwargs2)
			except TypeError:
				result=stone(path,*sling_args,**sling_kwargs)

		if result is not None:
			path_result=(path,result)
			if not stream_results: results+=[path_result]
			#if cache_writer:
			if cache_writer:
				#cache_writer.write(path_result) # when using jsonlines
				#try:
				#jsonl=json.dumps(path_result)
				jsonl=json.dumps(path_result,ignore_nan=True)
				cache_writer.write(jsonl + '\n')
				#except:
				#	print "!! could not write to results file!"
		#################################################
		#print('>> Clone #%s slings %s at #%s of %s %s enemies!' % (str(rank).zfill(zlen_rank),stone_name,str(i+1).zfill(zlen),pronoun,num_paths))
		#print('>> Clone #%s slings %s at Target #%s (of %s)' % (str(rank).zfill(zlen_rank),stone_name,str(i+1).zfill(zlen),num_paths))
		if not progress_bar: print('>> Clone #%s -- slings --> Target #%s / %s' % (str(rank).zfill(zlen_rank),str(i+1).zfill(zlen),num_paths))
	if cache_writer: cache_writer.close()

	# Gather the results
	#RESULTS = comm.gather(results, root=0)

	# If I am the seed process again
	if rank == 0:
		t3 = dt.now()
		print('>> SLINGSHOT: Finished parsing in %s seconds.' % (t3-t1).total_seconds())

		# Save results...
		if save_results and results_dir:
			# Make dir...
			if not os.path.exists(results_dir): os.makedirs(results_dir)

			# Copy pathlist
			results_fnfn_pathlist = os.path.join(results_dir,'pathlist.txt')
			results_fnfn_metadata = os.path.join(results_dir,'metadata.txt')
			save_results_pathlist(results_fnfn_pathlist,results_fnfn_metadata,all_paths,path_source)

			# Stream-save TSV
			results_fnfn_txt = os.path.join(results_dir,'results.txt')
			if save_txt: save_results_txt(results_fnfn_txt,cache_path,txt_maxcols)

		# Exit
		t4 = dt.now()
		print('>> SLINGSHOT: Finished everything in %s seconds!' % (t4-t1).total_seconds())




def save_results_pathlist(results_fnfn_pathlist,results_fnfn_metadata,paths,path_source):
	with open(results_fnfn_pathlist,'w') as of:
		for (path,run) in paths:
			try:
				of.write(str(path)+'\n')
			except TypeError:
				of.write(str(path.addr) + '\n')

	path_pathlists = CONFIG.get('PATH_PATHLISTS','')
	pathlist_path_source = os.path.join(path_pathlists,path_source if path_source else '')

	if not path_source:
		pass
	elif os.path.exists(path_source) and is_csv(path_source):
		from shutil import copyfile
		copyfile(path_source, results_fnfn_metadata)
	elif os.path.exists(pathlist_path_source) and is_csv(path_source):
		from shutil import copyfile
		copyfile(path_source, results_fnfn_metadata)



def load_stone_in_sling(path_sling,stone_name, exts=['','.py','.ipynb']):
	if not path_sling or not stone_name:
		print('!! sling or stone not specified')
		return
	elif not os.path.exists(path_sling):
		new_path=None
		for ext in exts:
			abs_path_sling_ext=os.path.join(CONFIG['PATH_SLINGS'],path_sling+ext)
			#print(abs_path_sling_ext)
			if os.path.exists(abs_path_sling_ext):
				new_path=abs_path_sling_ext
				break
		if not new_path:
			print("!!",path_sling,"does not exist")
			return
		path_sling=new_path

	if path_sling.endswith('.py'):
		try:
			import importlib.util
			spec = importlib.util.spec_from_file_location("sling", path_sling)
			sling = importlib.util.module_from_spec(spec)
			spec.loader.exec_module(sling)
		except ImportError:
			import imp
			sling = imp.load_source('sling', path_sling)


		stone = getattr(sling,stone_name)
		return stone

	elif path_sling.endswith('.ipynb'):
		import nbimporter
		nbimporter.options['only_defs'] = CONFIG.get('NBIMPORTER_ONLY_DEFS',False)

		ppath,pfn = os.path.split(path_sling)
		pname,pext = os.path.splitext(pfn)

		NBL = nbimporter.NotebookLoader(path=[ppath])
		sling = NBL.load_module(pname)
		stone = getattr(sling,stone_name)
		return stone

	elif path_sling.endswith('.R'):
		from rpy2.robjects import r as R
		# load all source
		with open(path_sling) as f:
			code=f.read()

			R('library(RJSONIO)')
			rfunc = R(code)
			#print('done!')
			stone = lambda _path: rconvert(rfunc(_path))
			return stone




def get_all_paths_from_folder(rootdir,ext='.txt'):
	for root, subdirs, files in os.walk(rootdir,followlinks=True):
		for fn in files:
			if fn.endswith(ext):
				yield os.path.join(root,fn)




def get_paths_from_csv(_fnfn,path_key=PATH_KEY,path_ext=PATH_EXT,path_prefix='',path_suffix='',sep='\t'):
	paths=[]
	#with codecs.open(_fnfn,encoding='utf-8') as pf:
	if not path_key: path_key=DEFAULT_PATH_KEY
	with open(_fnfn) as pf:
		reader=csv.DictReader(pf,delimiter=sep)
		for dx in reader:
			path=dx.get(path_key,'')
			if not path: continue
			if path_prefix: path=os.path.join(path_prefix,path)
			if path_suffix: path=path+path_suffix
			if path: paths+=[path]
	return paths

def is_csv(_fnfn,sep='\t'):
	if not os.path.exists(_fnfn): return False
	if os.path.isdir(_fnfn): return False
	with open(_fnfn) as pf:
		first_line=pf.readline()
		return sep in first_line

def get_paths_from_pathlist(_fnfn,sep='\t',path_key=PATH_KEY,path_prefix='',path_suffix=''):
	_fnfn_is_csv=is_csv(_fnfn,sep=sep)
	if _fnfn_is_csv:
		return get_paths_from_csv(_fnfn,path_key=path_key,sep=sep,path_prefix=path_prefix,path_suffix=path_suffix)
	else:
		with open(_fnfn) as pf:
			paths=[line.strip() for line in pf]
			paths=[x for x in paths if x]
			if path_prefix: paths=[os.path.join(path_prefix,x) for x in paths]
			if path_suffix: paths=[path+path_suffix for x in paths]
			return paths



def load_paths(path_source,path_ext,limit,shuffle_paths,path_key=PATH_KEY,path_prefix='',path_suffix=''):
	paths=None
	if path_source:
		if os.path.isdir(path_source):
			paths=list(get_all_paths_from_folder(path_source,path_ext)) if path_source else None
		elif os.path.exists(path_source):
			paths=get_paths_from_pathlist(path_source,path_key=path_key,path_prefix=path_prefix,path_suffix=path_suffix)
		else:
			path_pathlists = CONFIG.get('PATH_PATHLISTS','')
			pathlist_path_source = os.path.join(path_pathlists,path_source)
			if os.path.exists(pathlist_path_source):
				paths=get_paths(pathlist_path_source)
	#paths=[p for p in paths if os.path.exists(p)]
	if not paths:
		print('!! no paths given or found at %s' % path_source if path_source else '')
		return
	paths=sorted(list(set(paths)))
	paths=paths[:limit]
	if shuffle_paths:
		random.shuffle(paths)
	return paths




def save_results_txt(results_fnfn_txt,path_cache,txt_maxcols=TXT_MAXCOLS):
	now=time.time()

	# First find KEYS
	KEYS=set()
	if txt_maxcols: Count=Counter()

	for path,result in stream_results(path_cache,flatten=True):
		if hasattr(result,'keys'):
			if txt_maxcols:
				Count.update(list(result.keys()))
			else:
				KEYS=set(result.keys())
	if txt_maxcols: KEYS={x for x,y in Count.most_common(txt_maxcols)}

	then,now=now,time.time()
	print('>> save_txt: found keys in %ss' % int(now-then))
	print(KEYS)

	if KEYS:
		# Then loop again to write
		header=['_path']+sorted([six.text_type(x) for x in KEYS])
		with codecs.open(results_fnfn_txt,'w',encoding='utf-8') as results_f_txt:#, jsonlines.open(results_fnfn_json) as reader:
			results_f_txt.write('\t'.join(header) + '\n')
			#for path,result in reader.iter(skip_invalid=True):
			for path,result in stream_results(path_cache,flatten=True):
				result['_path']=path
				orow=[six.text_type(result.get(h,'')) for h in header]
				results_f_txt.write('\t'.join(orow) + '\n')
			print('>> saved:',results_fnfn_txt)
			then,now=now,time.time()
			print('>> save_txt: saved in %ss' % int(now-then))



def rconvert(robj):
	from rpy2.robjects import r as R
	from rpy2.robjects import pandas2ri
	pandas2ri.activate()
	rjson=R['toJSON'](robj, collapse=' ')
	rjson_str = pandas2ri.ri2py(rjson)[0]
	pyobj = json.loads(rjson_str)
	sys.stdout.flush()
	return pyobj


### Loading results


def stream_results(path_cache,ext='.jsonl',flatten=False):
	if 'jsonl' in os.path.basename(path_cache).split('.'):
		for path,data in stream_jsonl(path_cache,flatten=flatten):
			#if '.ipynb' in path: continue
			yield (path,data)
	else:
		for fn in sorted(os.listdir(path_cache)):
			if (not fn.endswith(ext) and not fn.endswith(ext+'.gz')): continue
			fnfn=os.path.join(path_cache,fn)
			print('>> streaming:',fnfn,'...')
			for path,data in stream_jsonl(fnfn,flatten=flatten):
				#if '.ipynb' in path: continue
				yield (path,data)


def stream_jsonl(fn,flatten=False):
	#from xopen import xopen
	import ujson as json
	#with xopen(fn) as f:
	with open(fn) as f:
		for ln in f:
			path,data=json.loads(ln)
			if not flatten or type(data)!=list:
				yield (path,data)
			else:
				for i,data_x in enumerate(data):
					yield (path if not i else '',data_x)


def writegen(fnfn,generator,header=None,args=[],kwargs={}):
	if 'jsonl' in fnfn.split('.'): return writegen_jsonl(fnfn,generator,args=args,kwargs=kwargs)
	with codecs.open(fnfn,'w',encoding='utf-8') as of:
		for i,dx in enumerate():
			if not header: header=sorted(dx.keys())
			if not i: of.write('\t'.join(header) + '\n')
			of.write('\t'.join([six.text_type(dx.get(h,'')) for h in header]) + '\n')
	print('>> saved:',fnfn)
