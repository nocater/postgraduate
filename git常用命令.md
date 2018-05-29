# Git 常用命令
## git diff
git diff <filename> 比较*工作区*和*暂存区*的文件  
git diff --cached <filename> 比较*暂存区*和版本库的文件  
git diff --staged <filename> 比较*暂存区*和版本库的文件  
git diff HEAD 比较*工作区*和*版本库*的文件
patch 恢复操作

## git clone
git clone <git_url> master <local_floder> 克隆项目  
git colne <git_url> -b 远程分支 <local_floder> 拉取特定分支  

## git branch
git branch new_branch 创建新的分支  
git branch -l 显示所有分支  
git branch -v 显示所有分支及最后提交信息  
git branch -d branch 删除分支  
git branch -D 强制删除未合并的分支  

## git status    
git status -s 简要显示状态信息

## git add  
git add .  提交所有到*暂存区*  
git add -u 将被追踪的文件全部提交到*暂存区*  
git add -A 将改动和新加文件提交到*暂存区*  
git add -f 强制提交  
git add -i 交互提交  

## git checkout
git checkout 汇总显示*工作区*、*暂存区*与HEAD的差异[显示与origin/master一致与否]  
git checkout [<brach\>] 切换分支  
**git checout <file> 目标是工作区的文件**  
git checkout [-q] [<commit>] [--] <paths>...  使用<commit\>版本覆盖工作区 如果省略<commit\>则指定*暂存区*进行覆盖  
git checkout – filename 撤销git .add <filename> 命令[使用*暂存区*文件覆盖*工作区*修改]  
git checkout branch – filename 将branch所指向的提交中的filename替换*暂存区*和*工作区*中相应的文件。
git checkout . 重置*工作区*，使用*暂存区*进行覆盖  
git checkout HEAD . 重置*工作区*，使用*版本库*进行覆盖

## git reset
<paths\> 有：重置指定路径文件 无：重置引用  
git reset [HEAD] 重置*暂存区*  
git reset HEAD --hard 使用*版本库*重置*暂存区*和*工作区*  
git reset HEAD --soft 仅重置引用，不重置*暂存区*和*工作区*
git reset [-mixed] HEAD^ 回退一次引用，并使用其覆盖*暂存区*

## git commit
git commit -amend
## git stash

## git tag

## git push

## git pull

## Tips
git rm --cache <filename\> 取消文件追踪  

### Markdown 的使用
\# 一级标题  \#\#二级标题  
两个空格换行  
\.\+\-序列使用   
\*斜体\* \_斜体\_  
\*\*粗体\*\*，\_\_粗体\_\_
