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
from tqdm import tqdm
import random,shutil
import inspect
DEFAULT_PATH_KEY='_path'
DEFAULT_EXT = 'txt'

PATH_KEY=CONFIG.get('PATH_KEY','')
PATH_EXT=CONFIG.get('PATH_EXT','').replace('.','')

if not PATH_KEY: PATH_KEY=DEFAULT_PATH_KEY
if not PATH_EXT: PATH_EXT=DEFAULT_PATH_EXT

TXT_MAXCOLS=25000
from smart_open import open
from tqdm import tqdm


def slingshot_single_shot(stone,path):
    return stone(path)

def get_paths_already_finished_from_cache(cache_path):
    for path,result in stream_results(cache_path,flatten=False,progress=False):
        if result:
            if type(path)==list: path=tuple(path)
            if type(path)==dict: path=tuple(sorted(list(path.items())))
            yield path

def now(now=None):
    import datetime as dt
    if not now:
        now=dt.datetime.now()
    elif type(now) in [int,float,str]:
        now=dt.datetime.fromtimestamp(now)

    return '{0}-{1}-{2} {3}:{4}:{5}'.format(now.year,str(now.month).zfill(2),str(now.day).zfill(2),str(now.hour).zfill(2),str(now.minute).zfill(2),str(now.second).zfill(2))

data_slingshot='.data_slingshot/'

def shoot(func, objects, path_src=None, savedir=None, overwrite=False, **kwargs):
    # init savedir
    funcname = func if type(func)==str else func.__name__
    if type(func)==str and (not path_src or not os.path.exists(path_src)):
        print('`func` must be real function object, or if a string, `path_src` must be specified.')
        return 

    # init
    if not savedir: savedir=f'_data_/{funcname}'
    # if not savedir: savedir=f'_data_/{funcname}/{now().split()[0]}'
    if os.path.exists(savedir) and overwrite: shutil.rmtree(savedir)
    if not os.path.exists(savedir): os.makedirs(savedir)
    if not 'resume' in kwargs: kwargs['resume']=True
    if not 'nosave' in kwargs: kwargs['nosave']=True
    path_code=f'{savedir}/func.py'
    path_objs=f'{savedir}/input.jsonl'

    # get source of function
    if type(func)!=str:
        func_txt=inspect.getsource(func)
        with open(path_code,'w') as of:
            of.write(func_txt)
    else:
        shutil.copyfile(path_src,path_code)

    # save objects
    with open(path_objs,'w') as of:
        for obj in tqdm(objects,total=len(objects),desc='[Slingshot] Preparing input'):
            of.write(json.dumps(obj)+'\n')

    # call command
    cmd=f'slingshot -code {path_code} -func {funcname} -pathlist {path_objs} -savedir {savedir} -resume'
    for k,v in kwargs.items():
        cmd+=f' -{k} {v}' if v is not True else f' -{k}'
    return cmd

def gather(savedir):
    return list(stream_results(os.path.join(savedir,'cache')))



def slingshot(path_sling=None,stone_name=None,stone_args=None,paths=None,llp_corpus=None,limit=None,path_source=None,stone=None,path_key=PATH_KEY,path_ext=None,path_prefix='',path_suffix='',cache_results=True,cache_path=None,save_results=True,results_dir=None,shuffle_paths=True,do_stream_results=True,save_txt=True,txt_maxcols=TXT_MAXCOLS,sling_args=[],sling_kwargs={},num_runs=1,oneshot=False,llp_pass_text=False,llp_method='',progress_bar=True,savecsv='',resume=False,overwrite=False,parallel=1,llp_pass_path='path'):
    """
    Main function
    """
    progress_bar=True# if not sbatch else False

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
            #print(llp,dir(llp))
            corpus = llp.load_corpus(llp_corpus)
            #print(corpus.path_metadata)
            #print(llp_corpus, corpus)
            #print('llp_pass_path',llp_pass_path)
            #print('llp_pass_text',llp_pass_text)
            #print('llp_method',llp_method)
            #print(corpus.texts())
            all_paths = [(text.addr if (llp_pass_text or llp_method) else getattr(text,llp_pass_path)) for text in corpus.texts()][:limit]
            #print(all_paths[:10])
            if shuffle_paths:
                import random
                random.shuffle(all_paths)
        except ImportError as e:
            print('!!',e)
            pass

    # Get paths!
    if not all_paths:
        all_paths = load_paths(path_source,path_ext,limit,shuffle_paths,path_key,path_prefix,path_suffix) if not paths else paths

    # Break if these weren't returned
    if not all_paths: return


    # Report back
    #print('>> [Slingshot] # of paths:',len(all_paths))

    # Cache stuff
    if not results_dir: results_dir=f'data_slingshot/{stone_name}'
    if not cache_path: cache_path=os.path.join(results_dir,'cache')
    if cache_results:
        if overwrite and os.path.exists(cache_path): delete_files_from_folder(cache_path)
        if overwrite and os.path.exists(results_dir): delete_files_from_folder(results_dir)
        if not os.path.exists(cache_path):
            try:
                os.makedirs(cache_path)
            except FileExistsError:
                pass
        if resume:
            paths_done = set(list(get_paths_already_finished_from_cache(cache_path)))
            print('\n>> [Slingshot] already finished %s of %s' % (len(paths_done),len(all_paths)))
            try:
                all_paths=list(set(all_paths)-paths_done)
            except TypeError:
                pass

        import random
        if shuffle_paths: random.shuffle(all_paths)

    ## de-tuple?
    all_paths = [
        dict(p) if (p and type(p)==tuple and set(len(x) for x in p)=={2}) else p
        for p in all_paths
    ]

    ## RUNS?
    # Multiply paths by runs
    all_paths = [(path,run+1) for path in all_paths for run in range(num_runs)]

    t1 = dt.now()
    if parallel < 2:
        paths=all_paths
        results=[]
        looper=all_paths if not progress_bar else tqdm(all_paths,file=sys.stdout,desc='Slingshot',position=0,ncols=100)
        if cache_results and cache_path:
            cache_fn = 'results.jsonl'
            cache_fnfn = os.path.join(cache_path,cache_fn)
            cache_writer = open(cache_fnfn, mode='a+',encoding='utf-8')
        rank = 0

    else:
        # Start MPI
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
        #if rank==0: print()
        #print('>> [Slingshot] initializing MPI with size %s and rank %s' % (size,rank))


        # Am I the seed process?
        if rank == 0:
            # Farm out paths to other processes
            segments = np.array_split(all_paths,size) if size>1 else [all_paths]
            print('>> [Slingshot] %s paths divided into %s segments' % (len(all_paths), len(segments)))

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
            cache_writer = open(cache_fnfn, mode='a+',encoding='utf-8')

        #progress_bar=True
        desc=f'>> [Slingshot] Compute worker {rank}'
        looper=paths if not progress_bar else tqdm(paths,file=sys.stdout,desc=desc,position=rank)



    # let's go! loop over the paths
    results=[]
    num_paths=len(paths)
    pronoun='their'
    zlen=len(str(num_paths))
    zlen_rank=3 #len(str(size))



    for i,(path,run) in enumerate(looper):
        #################################################
        # THIS IS WHERE THE STONE FITS INTO THE SLINGSHOT

        sling_kwargs2=dict(list(sling_kwargs.items()))
        #sling_kwargs2['results_dir']=results_dir
        if num_runs>1: sling_kwargs2['run']=run

        path_store = None

        if llp_pass_text and not llp_method:
            idx = path
            text = corpus.text(path)
            path_store = text.addr
            path = text.path_txt

        if llp_method:
            idx = path
            try:
                text = corpus.text(path)
                path = text.addr
                stone = getattr(text,llp_method)
            except AttributeError as e:
                print("!!",e)
                return
            result = stone(*sling_args,**sling_kwargs2)

        else:
            try:
                result=stone(path,*sling_args,**sling_kwargs2)
            except TypeError:
                result=stone(path,*sling_args,**sling_kwargs)

        if result is not None:
            path_store = path if not path_store else path_store
            path_result=(path_store,result)
            if not do_stream_results: results+=[path_result]
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
        #if not progress_bar: print('>> Clone #%s -- slings --> Target #%s / %s' % (str(rank).zfill(zlen_rank),str(i+1).zfill(zlen),num_paths))
    if cache_writer: cache_writer.close()

    # Gather the results
    if parallel>1: RESULTS = comm.gather(results, root=0)

    # If I am the seed process again
    if rank == 0:
        t3 = dt.now()
        print('>> [Slingshot] Finished parsing in %s seconds.' % (t3-t1).total_seconds())

        # Save results...
        if save_results and results_dir:
            # Make dir...
            if not os.path.exists(results_dir): os.makedirs(results_dir)

            # Copy pathlist
            results_fnfn_pathlist = os.path.join(results_dir,'pathlist.txt')
            results_fnfn_metadata = os.path.join(results_dir,'metadata.txt')
            save_results_pathlist(results_fnfn_pathlist,results_fnfn_metadata,all_paths,path_source)

            # Stream-save TSV
            results_fnfn_txt = os.path.join(results_dir,'results.txt') if not savecsv else savecsv
            if save_txt: save_results_txt(results_fnfn_txt,cache_path,txt_maxcols)

        # Exit
        t4 = dt.now()
        print('>> [Slingshot] Finished everything in %s seconds!' % (t4-t1).total_seconds())




def save_results_pathlist(results_fnfn_pathlist,results_fnfn_metadata,paths,path_source):
    with open(results_fnfn_pathlist,'w') as of:
        for (path,run) in paths:
            try:
                of.write(str(path)+'\n')
            except TypeError:
                of.write(str(path.addr) + '\n')

    path_pathlists = CONFIG.get('PATH_PATHLISTS','')
    pathlist_path_source = os.path.join(path_pathlists,path_source if path_source else '')

    #print(path_source,'!?!?!?')
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
    elif _fnfn.endswith('jsonl'):
        import json
        jsons=[]
        with open(_fnfn) as f:
            for ln in f:
                try:
                    d=json.loads(ln.strip())
                    jsons.append(d)
                except (ValueError,json.JSONDecodeError) as e:
                    pass
        return jsons
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
    #paths=sorted(list(set(paths)))
    paths=list(paths)
    paths=paths[:limit]
    if shuffle_paths:
        random.shuffle(paths)
    paths=[tuple(x) if type(x)==list else x for x in paths]
    paths=[tuple(sorted(list(x.items()))) if x==dict else x for x in paths]
    return paths




def save_results_txt(results_fnfn_txt,path_cache,txt_maxcols=TXT_MAXCOLS,sep='\t'):
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
    #print('>> save_txt: found %s keys in %ss' % (len(KEYS),int(now-then)))
    #print(list(KEYS)[:100],'...')

    if KEYS:
        # Then loop again to write
        header=['_path']+sorted([six.text_type(x) for x in KEYS])
        #with codecs.open(results_fnfn_txt,'w',encoding='utf-8') as results_f_txt:#, jsonlines.open(results_fnfn_json) as reader:
        with open(results_fnfn_txt,'w') as results_f_txt:
            results_f_txt.write(sep.join(header) + '\n')
            for path,result in stream_results(path_cache,flatten=True):
                result['_path']=path
                orow=[str(result.get(h,'')).replace(sep,' ').replace('\r\n',' ').replace('\r',' ').replace('\n',' ') for h in header]
                results_f_txt.write(sep.join(orow) + '\n')
            #print('>> saved:',results_fnfn_txt)
            then,now=now,time.time()
            #print('>> save_txt: saved in %ss' % int(now-then))



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


def stream_results(path_cache,ext='.jsonl',flatten=False,progress=True):
    if 'jsonl' in os.path.basename(path_cache).split('.'):
        for path,data in stream_jsonl(path_cache,flatten=flatten,progress=progress):
            #if '.ipynb' in path: continue
            yield (path,data)
    else:
        for fn in sorted(os.listdir(path_cache)):
            if (not fn.endswith(ext) and not fn.endswith(ext+'.gz')): continue
            fnfn=os.path.join(path_cache,fn)
            #print('>> streaming:',fnfn,'...')
            for path,data in stream_jsonl(fnfn,flatten=flatten,progress=progress):
                #if '.ipynb' in path: continue
                yield (path,data)


def stream_jsonl(fn,flatten=False,progress=True):
    #from xopen import xopen
    import ujson as json
    if progress: from tqdm import tqdm

    #with xopen(fn) as f:
    #print('>> [Slingshot] streaming:',fn,'...')
    if progress: num_lines = get_num_lines(fn)
    with open(fn) as f:
        looper = tqdm(f,total=num_lines,desc=f'[Slingshot] Reading {os.path.basename(fn)}') if progress else f
        for ln in looper:
            try:
                path,data=json.loads(ln)
            except ValueError:
                continue
            if not flatten or type(data)!=list:
                yield (path,data)
            else:
                for i,data_x in enumerate(data):
                    yield (path if not i else '',data_x)


def writegen(fnfn,generator,header=None,args=[],kwargs={}):
    if 'jsonl' in fnfn.split('.'): return writegen_jsonl(fnfn,generator,args=args,kwargs=kwargs)
    with open(fnfn,'w') as of:
        for i,dx in enumerate(generator()):
            if not header: header=sorted(dx.keys())
            if not i: of.write('\t'.join(header) + '\n')
            of.write('\t'.join([six.text_type(dx.get(h,'')) for h in header]) + '\n')
    print('>> saved:',fnfn)




def get_num_lines(filename):
    from smart_open import open

    def blocks(files, size=65536):
        while True:
            b = files.read(size)
            if not b: break
            yield b

    with open(filename, 'r', errors='ignore') as f:
        numlines=sum(bl.count("\n") for bl in blocks(f))

    return numlines





def iter_move(fn,force=False,prefix=''):
    if os.path.exists(fn):
        iter_fn=iter_filename(fn,force=force,prefix=prefix)
        iter_dir=os.path.dirname(iter_fn)
        if not os.path.exists(iter_dir): os.makedirs(iter_dir)
        shutil.move(fn,iter_fn)
        print(f'>> moved: {fn} --> {iter_fn}')

def iter_filename(fnfn,force=False,prefix='',inbetweener='_'):
    if os.path.exists(fnfn) or force:
        fndir,fn=os.path.split(fnfn)
        filename,ext = os.path.splitext(fn)
        fnum=2 if not force else 1
        maybe_fn=os.path.join(fndir, prefix + filename + ext)
        while os.path.exists(maybe_fn):
            fnum+=1
            maybe_fn=os.path.join(fndir, prefix + filename + inbetweener + str(fnum) + ext)
        fnfn = maybe_fn
    return fnfn




def delete_files_from_folder(path,toprint=True):
    if os.path.exists(path):
        for root, dirs, files in os.walk(path):
            for fn in files:
                fnfn=os.path.join(root, fn)
                os.unlink(fnfn)
                if toprint: print('>> deleted:',fnfn)
